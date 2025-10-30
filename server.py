# server.py
import os
import json
from dotenv import load_dotenv
from fastmcp import FastMCP
from rag import FileRAG
from github import Github
import subprocess


load_dotenv()

mcp = FastMCP(
    name="MCP-RAG-Agent",
    instructions="Custom MCP server."
)

rag = FileRAG()


@mcp.tool()
def list_directory(path: str) -> dict:
    if not os.path.isdir(path):
        return {"error": f"'{path}' is not valid dir."}
    files = os.listdir(path)
    return {"files": files}

@mcp.tool()
def read_file(path: str) -> dict:
    if not os.path.exists(path):
        return {"error": f"file '{path}' can't find."}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return {"content": content}

@mcp.tool()
def build_index_from_directory(path: str) -> dict:
    if not os.path.isdir(path):
        return {"error": f"'{path}' is not a valid directory."}

    file_dict = {}
    total_files = 0

    for fname in os.listdir(path):
        full_path = os.path.join(path, fname)
        if not os.path.isfile(full_path):
            continue

        try:
            ext = fname.lower().split(".")[-1]
            text = ""

            if ext == "pdf":
                text = rag.pdf_to_text(full_path)
            elif ext in ["py", "md", "txt", "json", "yaml", "yml"]:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            else:
                continue 

            if text.strip():
                file_dict[fname] = text
                total_files += 1
            else:
                print(f"[SKIP] {fname}: empty text")

        except Exception as e:
            print(f"[WARN] {fname} failed: {e}")
            continue

    if not file_dict:
        return {"error": "No valid files found for indexing."}

    rag.build_index(file_dict)
    return {"message": f"{total_files} files indexed successfully."}


@mcp.tool()
def build_index_from_github(url: str) -> dict:
    try:
        if "github.com" not in url:
            return {"error": "is not valid GitHub URL."}

        repo_path = url.split("github.com/")[-1].split("/tree")[0].strip("/")
        g = Github(os.getenv("GITHUB_TOKEN"))
        repo = g.get_repo(repo_path)

        file_dict = {}
        stack = repo.get_contents("")

        while stack:
            file_content = stack.pop(0)
            if file_content.type == "dir":
                stack.extend(repo.get_contents(file_content.path))
            elif file_content.type == "file":
                fname = file_content.path
                try:
                    if fname.lower().endswith((".py", ".c", ".md", ".txt", ".json", ".yaml", ".yml")):
                        text = file_content.decoded_content.decode("utf-8", errors="ignore")
                        if text.strip():
                            file_dict[fname] = text
                except Exception as e:
                    print(f"[WARN] {fname} fail: {e}")
                    continue

        if not file_dict:
            return {"error": f"'{repo_path}' can't find."}

        rag.build_index(file_dict)
        return {"message": f"GitHub: '{repo_path}', {len(file_dict)} indexing complete."}

    except Exception as e:
        return {"error": f"GitHub indexing fail: {str(e)}"}


@mcp.tool()
def search_in_index(query: str, top_k: int = 3) -> dict:
    try:
        results = rag.search(query, top_k=top_k)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def write_file(path: str, content: str) -> dict:
    try:
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"message": f"file save: {path}"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def git_commit_and_push(repo_path: str, commit_message: str, remote_url: str) -> dict:
    try:
        if not os.path.isdir(repo_path):
            return {"error": f"{repo_path} is not a valid directory."}

        username = os.getenv("GITHUB_USERNAME")
        token = os.getenv("GITHUB_TOKEN")
        if not username or not token:
            return {"error": "GitHub Authentication info missing."}

        auth_url = remote_url.replace(
            "https://",
            f"https://{username}:{token}@"
        )

        subprocess.run(["git", "-C", repo_path, "add", "."], check=True)
        subprocess.run(["git", "-C", repo_path, "commit", "-m", commit_message], check=False)
        subprocess.run(["git", "-C", repo_path, "push", auth_url, "--force"], check=True)

        return {"message": f"✅ '{commit_message}' pushed to {remote_url}!"}

    except subprocess.CalledProcessError as e:
        return {"error": f"Git command error: {e}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000,
        path="/mcp",
    )
