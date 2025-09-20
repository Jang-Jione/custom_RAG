import json, os, sys, asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rag import FileRAG
from llm_api import query_gpt

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

        # 📂 파일 로딩
        file_dict = {}
        resp = await session.call_tool("list_directory", {"path": target_dir})
        files = resp.content[0].text.split("\n")

        for fname in files:
            fname = fname.strip()
            if fname.endswith((".py", ".md", ".txt")):
                # 이미 list_directory가 상대경로로 반환하므로 target_dir 붙일 필요 없음
                file_resp = await session.call_tool("read_file", {"path": os.path.join(target_dir, fname)})
                content = file_resp.content[0].text
                file_dict[fname] = content

        print(f"총 {len(file_dict)}개 파일 로드 완료")

        # 📌 RAG 인덱스 구축
        rag = FileRAG()
        rag.build_index(file_dict)

        # 🔄 대화형 모드
        while True:
            user_input = input("\n❯ ").strip()
            if user_input.lower() == "exit":
                print("세션 종료, DB 삭제 완료")
                del rag
                break

            # Git 명령어 처리
            if user_input.lower().startswith("status"):
                result = await session.call_tool("git_status", {"cwd": target_dir})
                print(result.content[0].text)
                continue

            if user_input.lower().startswith("commit"):
                # commit "메시지" 형식
                parts = user_input.split(" ", 1)
                if len(parts) == 2:
                    msg = parts[1].strip('"')
                    result = await session.call_tool("git_commit", {"message": msg, "cwd": target_dir})
                    print(result.content[0].text)
                else:
                    print("❌ commit 메시지를 입력하세요. 예: commit \"update docs\"")
                continue

            if user_input.lower().startswith("push"):
                result = await session.call_tool("git_push", {"cwd": target_dir})
                print(result.content[0].text)
                continue

            # GPT 분석 요청
            related_files = rag.search(user_input, top_k=3)
            context = "\n\n".join([f"[{fname}]\n{file_dict[fname]}" for fname in related_files])
            prompt = f"""
            You are a coding assistant.  

            Here is the project context (code files):

            {context}

            ---

            User request: {user_input}

            ---

            Task:
            - Use the given context (code) as the primary source of information.
            - Answer the user request clearly and accurately.
            - If the request is about improvements, suggest concrete and actionable improvements to the code.
            - If the request is about explanation, explain the relevant parts of the code step by step.
            - Answer in Korean.
            """
            analysis = query_gpt(prompt)
            print(analysis)

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"분석 대상 디렉토리: {target_dir}")
    asyncio.run(run_session(target_dir))
