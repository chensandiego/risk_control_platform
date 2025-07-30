
from github import Github
import os

GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

def get_github_client():
    """Create and return a GitHub client."""
    if not GITHUB_ACCESS_TOKEN:
        return {"error": "GitHub access token not configured."}
    return Github(GITHUB_ACCESS_TOKEN)

def list_user_repos():
    """List all repositories for the authenticated user."""
    g = get_github_client()
    if isinstance(g, dict) and "error" in g:
        return g
    try:
        repos = []
        for repo in g.get_user().get_repos():
            repos.append(repo.full_name)
        return {"repositories": repos}
    except Exception as e:
        return {"error": str(e)}

def get_repo_file_content(repo_name: str, file_path: str):
    """Get the content of a file from a repository."""
    g = get_github_client()
    if isinstance(g, dict) and "error" in g:
        return g
    try:
        repo = g.get_repo(repo_name)
        content = repo.get_contents(file_path)
        return content.decoded_content
    except Exception as e:
        return {"error": str(e)}
