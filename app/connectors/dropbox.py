
import dropbox
import os

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")

def get_dropbox_client():
    """Create and return a Dropbox client."""
    if not DROPBOX_ACCESS_TOKEN:
        return {"error": "Dropbox access token not configured."}
    return dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

def list_dropbox_files(path: str = ""):
    """List files in a Dropbox folder."""
    dbx = get_dropbox_client()
    if isinstance(dbx, dict) and "error" in dbx:
        return dbx
    try:
        return dbx.files_list_folder(path)
    except dropbox.exceptions.ApiError as e:
        return {"error": str(e)}

def download_dropbox_file(file_path: str):
    """Download a file from Dropbox."""
    dbx = get_dropbox_client()
    if isinstance(dbx, dict) and "error" in dbx:
        return dbx
    try:
        _, res = dbx.files_download(path=file_path)
        return res.content
    except dropbox.exceptions.ApiError as e:
        return {"error": str(e)}
