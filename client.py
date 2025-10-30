import os, json, uuid, requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MCP_URL = "http://127.0.0.1:8000/mcp"
session_id: str | None = None 


def init_mcp_session():
    global session_id
    if session_id is not None:
        return session_id  

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "initialize",
        "params": {
            "protocolVersion": 1,
            "capabilities": {"tools": True},
            "clientInfo": {"name": "local-client", "version": "0.1"}
        }
    }
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json"
    }

    r = requests.post(MCP_URL, json=payload, headers=headers, timeout=10)
    r.raise_for_status()

    sid = r.headers.get("Mcp-Session-Id")
    if not sid:
        raise RuntimeError("Session ID issue.")
    session_id = sid
    print(f"✅ MCP Sesseion Build!: {session_id}")

    notify = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {}
    }
    headers["Mcp-Session-Id"] = sid
    requests.post(MCP_URL, json=notify, headers=headers, timeout=5)
    return sid


def call_mcp_tool(name: str, args: dict):
    global session_id
    if session_id is None:
        init_mcp_session()

    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {"name": name, "arguments": args}
    }

    print("\n=== [JSON-RPC Request] ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False)) 
    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "Mcp-Session-Id": session_id
    }

    with requests.post(MCP_URL, json=payload, headers=headers, stream=True, timeout=100) as r:
        if r.status_code != 200:
            raise RuntimeError(f"ERROR: {r.status_code} {r.text}")

        data_lines = []
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if not data_lines:
            return {"error": "response is empty."}

        raw = "\n".join(data_lines)
        try:
            outer = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

        if isinstance(outer, dict) and "result" in outer:
            content = outer["result"].get("content", [])
            if content and "text" in content[0]:
                raw_text = content[0]["text"]
                try:
                    data = json.loads(raw_text)
                    if isinstance(data, str):
                        data = json.loads(data)
                    return data
                except json.JSONDecodeError:
                    return {"raw": raw_text}

def chat_with_mcp():
    init_mcp_session()

    messages = [
        {
            "role": "system",
            "content": "너는 로컬 RAG MCP 서버의 도구를 활용할 수 있는 AI야. 사용자가 파일 검색, 파일 작성 등의 작업을 요청하면 MCP 도구를 호출해야 해."
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "로컬 폴더의 파일 목록을 가져옵니다.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "build_index_from_directory",
                "description": "폴더 내 파일을 인덱싱합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "build_index_from_github",
                "description": "깃허브 URL을 RAG 인덱스로 구축합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_in_index",
                "description": "RAG 인덱스에서 쿼리를 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "git_commit_and_push",
                "description": "로컬 폴더의 변경사항을 커밋하고 지정한 원격 저장소 URL로 푸시합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "로컬 Git 저장소의 경로"
                        },
                        "commit_message": {
                            "type": "string",
                            "description": "커밋 메시지"
                        },
                        "remote_url": {
                            "type": "string",
                            "description": "원격 저장소의 HTTPS 또는 SSH URL (예: https://github.com/user/repo.git)"
                        }
                    },
                    "required": ["repo_path", "commit_message", "remote_url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "지정된 경로에 파일을 작성합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            }
        }
    ]

    print("Session start (escape: quit, exit)\n")
    while True:
        user_input = input("💬 Prompt: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("MCP session finish!")
            break

        messages.append({"role": "user", "content": user_input})

        chat_kwargs = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "tool_choice": "auto",
            "tools": tools
        }

        res = client.chat.completions.create(**chat_kwargs)
        msg = res.choices[0].message

        if not msg.tool_calls:
            print("", msg.content)
            messages.append({"role": "assistant", "content": msg.content})
            continue

        for call in msg.tool_calls:
            fn_name = call.function.name
            fn_args = json.loads(call.function.arguments)
            print(f"[MCP request] {fn_name}({fn_args})")
            tool_output = call_mcp_tool(fn_name, fn_args)
            # print(f"[MCP response] {tool_output}")
            print(f"[MCP response] success")

            messages += [
                {"role": "assistant", "content": None, "tool_calls": [call.model_dump()]},
                {"role": "tool", "tool_call_id": call.id, "name": fn_name,
                 "content": json.dumps(tool_output, ensure_ascii=False)}
            ]
        
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        final_msg = res.choices[0].message
        print("🧠 GPT:", final_msg.content)
        messages.append({"role": "assistant", "content": final_msg.content})


if __name__ == "__main__":
    chat_with_mcp()
