
import gitlab
import os

GITLAB_URL = os.getenv("GITLAB_URL")
GITLAB_PRIVATE_TOKEN = os.getenv("GITLAB_PRIVATE_TOKEN")

def get_gitlab_client():
    """Create and return a GitLab client."""
    if not all([GITLAB_URL, GITLAB_PRIVATE_TOKEN]):
        return {"error": "GitLab URL or private token not configured."}
    return gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_PRIVATE_TOKEN)

def list_projects():
    """List all projects for the authenticated user."""
    gl = get_gitlab_client()
    if isinstance(gl, dict) and "error" in gl:
        return gl
    try:
        projects = []
        for project in gl.projects.list():
            projects.append(project.name_with_namespace)
        return {"projects": projects}
    except Exception as e:
        return {"error": str(e)}

def get_project_file_content(project_id: int, file_path: str, ref: str = "main"):
    """Get the content of a file from a project."""
    gl = get_gitlab_client()
    if isinstance(gl, dict) and "error" in gl:
        return gl
    try:
        project = gl.projects.get(project_id)
        file = project.files.get(file_path=file_path, ref=ref)
        return file.decode()
    except Exception as e:
        return {"error": str(e)}
