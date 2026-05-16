import os
import json
from urllib.request import Request, urlopen
from urllib.error import URLError
import gdown


def _download_drive_usercontent(file_id: str, output_path: str) -> None:
    url = f"https://drive.usercontent.google.com/download?id={file_id}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, open(output_path, "wb") as f:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            raise RuntimeError("Google Drive returned an HTML page instead of the file.")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def download_gdown_file(file_id: str, output_path: str, quiet: bool = False) -> str:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    try:
        result = gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=quiet)
        if result is None:
            raise RuntimeError("gdown returned no output path.")
    except (gdown.exceptions.FileURLRetrievalError, RuntimeError, URLError) as exc:
        if not quiet:
            print(f"gdown could not resolve the Drive URL ({exc}); trying direct download.")
        _download_drive_usercontent(file_id, output_path)
    return output_path


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)