from mcp.server.fastmcp import FastMCP
import os
import subprocess
import requests

mcp = FastMCP("PythonFS+Git+GitHub")


def is_github_url(path: str) -> bool:
    return path.startswith("https://github.com/")

def github_api_url(repo_url: str) -> str:
    repo_url = repo_url.replace(".git", "")
    parts = repo_url.split("/")
    if len(parts) < 5:
        raise ValueError("Invalid GitHub URL format.")
    owner, repo = parts[3], parts[4]
    subpath = "/".join(parts[5:]) if len(parts) > 5 else ""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    if subpath:
        api_url += f"/{subpath}"
    return api_url

def fetch_github_directory(api_url: str):
    resp = requests.get(api_url)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API Error: {resp.status_code} {resp.text}")
    items = resp.json()
    file_list = []
    for item in items:
        if item["type"] == "file":
            if item["name"].endswith((".py", ".md", ".txt", ".pdf")):
                file_list.append(item["download_url"])
        elif item["type"] == "dir":
            file_list.extend(fetch_github_directory(item["url"]))
    return file_list

def fetch_github_file(download_url: str) -> str:
    resp = requests.get(download_url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch file: {download_url}")
    return resp.text


@mcp.tool()
def list_directory(path: str = ".") -> str:
    try:
        if is_github_url(path):
            print(f"[GITHUB] listing files from {path}")
            api_url = github_api_url(path)
            file_list = fetch_github_directory(api_url)
        else:
            print(f"[LOCAL] listing files from {os.path.abspath(path)}")
            EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "node_modules", "dist", "build"}
            file_list = []
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for f in files:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, path)
                    if f.endswith((".py", ".md", ".txt", ".pdf")):
                        file_list.append(rel_path)
        return "\n".join(file_list)
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def read_file(path: str) -> str:
    try:
        if path.startswith("https://raw.githubusercontent.com/"):
            return fetch_github_file(path)
        elif is_github_url(path):
            # https://github.com/user/repo/blob/main/file.py 형태 처리
            if "/blob/" in path:
                raw_url = (
                    path.replace("github.com", "raw.githubusercontent.com")
                        .replace("/blob/", "/")
                )
                return fetch_github_file(raw_url)
            else:
                # https://github.com/user/repo/... → API 변환 후 download_url 찾기
                api_url = github_api_url(path)
                resp = requests.get(api_url)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and "download_url" in data:
                        return fetch_github_file(data["download_url"])
                raise RuntimeError(f"Invalid GitHub file path: {path}")
        else:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def write_file(path: str, content: str) -> str:
    try:
        if is_github_url(path):
            return "GitHub 원격 레포에는 직접 쓰기가 불가능합니다."
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"{path} 저장 완료"
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def git_status(cwd: str = ".") -> str:
    try:
        result = subprocess.run(["git", "-C", cwd, "status"], capture_output=True, text=True, check=True)
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def git_commit(message: str, cwd: str = ".") -> str:
    try:
        subprocess.run(["git", "-C", cwd, "add", "."], check=True)
        result = subprocess.run(
            ["git", "-C", cwd, "commit", "-m", message],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def git_push(cwd: str = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "push"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
