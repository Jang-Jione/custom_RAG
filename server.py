# server.py
from mcp.server.fastmcp import FastMCP
import os
import subprocess

mcp = FastMCP("PythonFS+Git")

# =========================
# Filesystem Tools
# =========================
@mcp.tool()
def list_directory(path: str = ".") -> str:
    """지정된 디렉토리와 하위 폴더까지 파일/폴더 리스트 (불필요한 폴더 제외)"""
    EXCLUDE_DIRS = {".venv", "__pycache__", ".git", "node_modules", "dist", "build"}
    file_list = []
    for root, dirs, files in os.walk(path):
        # 불필요한 디렉토리 제외
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            file_list.append(os.path.relpath(os.path.join(root, f), path))
    return "\n".join(file_list)

@mcp.tool()
def read_file(path: str) -> str:
    """파일 내용을 문자열로 반환"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# Git Tools
# =========================
@mcp.tool()
def git_status(cwd: str = ".") -> str:
    """git status 출력"""
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "status"],
            capture_output=True, text=True, check=True
        )
        return result.stdout
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
def git_commit(message: str, cwd: str = ".") -> str:
    """변경된 파일을 모두 추가하고 commit"""
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
    """원격 저장소로 push"""
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
