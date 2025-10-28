import json, os, sys, asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rag import FileRAG
from llm_api import query_gpt


def is_github_url(path: str) -> bool:
    return path.startswith("https://github.com/") or path.startswith("https://raw.githubusercontent.com/")


async def run_session(target_dir: str):
    with open("mcp_server_config.json") as f:
        config = json.load(f)["mcpServers"]["filesystem_git"]

    server_params = StdioServerParameters(
        command=config["command"],
        args=config["args"]
    )

    async with AsyncExitStack() as stack:
        stdio, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(stdio, write))
        await session.initialize()

        print(f"[INFO] 분석 대상: {target_dir}")

        file_dict = {}

        # MCP 서버에 list_directory 요청
        resp = await session.call_tool("list_directory", {"path": target_dir})
        files_raw = resp.content[0].text.strip()
        if not files_raw or files_raw.startswith("Error:"):
            print(f"[ERROR] 디렉토리 탐색 실패: {files_raw}")
            return
        files = files_raw.split("\n")

        for fname in files:
            fname = fname.strip()
            if not fname:
                continue

            if fname.endswith((".py", ".md", ".txt", ".pdf")):
                if is_github_url(fname):
                    path = fname
                else:
                    path = os.path.join(target_dir, fname)

                if fname.lower().endswith(".pdf"):
                    file_dict[fname] = path
                else:
                    try:
                        file_resp = await session.call_tool("read_file", {"path": path})
                        content = file_resp.content[0].text
                        if content.startswith("Error:"):
                            print(f"{fname} 읽기 실패: {content}")
                        else:
                            file_dict[fname] = content
                    except Exception as e:
                        print(f"{fname} 읽기 중 오류 발생: {e}")

        print(f"\n총 {len(file_dict)}개 파일 로드 완료")
        print("로드된 파일:", list(file_dict.keys())[:10])

        rag = FileRAG()
        rag.build_index(file_dict)

        while True:
            user_input = input("\n❯ ").strip()
            if user_input.lower() == "exit":
                print("세션 종료, DB 삭제 완료")
                del rag
                break

            if user_input.lower().startswith("status"):
                result = await session.call_tool("git_status", {"cwd": target_dir})
                print(result.content[0].text)
                continue

            if user_input.lower().startswith("commit"):
                parts = user_input.split(" ", 1)
                if len(parts) == 2:
                    msg = parts[1].strip('"')
                    result = await session.call_tool(
                        "git_commit", {"message": msg, "cwd": target_dir}
                    )
                    print(result.content[0].text)
                else:
                    print("commit 메시지를 입력하세요. 예: commit \"update docs\"")
                continue

            if user_input.lower().startswith("push"):
                result = await session.call_tool("git_push", {"cwd": target_dir})
                print(result.content[0].text)
                continue

            # ------------------ 파일 저장 ------------------
            if user_input.lower().startswith("save"):
                parts = user_input.split(" ", 1)
                if len(parts) == 2:
                    fname = parts[1].strip()
                    print(f"저장할 파일명: {fname}")
                    print("붙여넣을 코드 내용을 입력하세요 (끝내려면 빈 줄 입력):")
                    buffer = []
                    while True:
                        line = input()
                        if not line.strip():
                            break
                        buffer.append(line)
                    content = "\n".join(buffer)
                    result = await session.call_tool(
                        "write_file",
                        {"path": os.path.join(target_dir, fname), "content": content},
                    )
                    print(result.content[0].text)
                else:
                    print("저장할 파일명을 입력하세요. 예: save test.py")
                continue

            # ------------------ GPT 분석 ------------------
            related_files = rag.search(user_input, top_k=3)
            context_parts = []
            for fname in related_files:
                for doc_fname, doc_text in rag.docs:
                    if doc_fname == fname:
                        context_parts.append(f"[{fname}]\n{doc_text}")
                        break

            context = "\n\n".join(context_parts)
            prompt = f"""
            You are a coding assistant.

            Here is the project context (code/files):

            {context}

            ---

            User request: {user_input}

            ---

            Task:
            - Use the given context (code/files) as the primary source of information.
            - Answer the user request clearly and accurately.
            - If the request is about improvements, suggest concrete and actionable improvements to the code.
            - If the request is about explanation, explain the relevant parts of the code step by step.
            - Answer in Korean.
            """
            analysis = query_gpt(prompt)
            print(analysis)


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    asyncio.run(run_session(target_dir))
