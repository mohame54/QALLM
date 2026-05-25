import os
from huggingface_hub import HfApi, hf_hub_download


def load_hf_api():
    return HfApi(token=os.getenv("HF_TOKEN"))


def upload_file_paths_to_hf(pathes):
    api = load_hf_api()
    repo_id = os.getenv("HF_REPO_ID")
    for pth in pathes:
         api.upload_file(
            path_or_fileobj= pth,
            path_in_repo =pth,
            repo_id=repo_id,
            repo_type="model",
        )


def upload_folder_to_hf(
    local_dir: str,
    path_in_repo: str = "",
    repo_id: str | None = None,
    commit_message: str = "Upload model checkpoint",
) -> str:
    """Push an entire local directory to a HuggingFace Hub model repo.

    Args:
        local_dir: Local folder to upload (e.g. ``output_dir``).
        path_in_repo: Sub-folder inside the HF repo (``""`` = repo root).
        repo_id: HF repo id; falls back to ``HF_REPO_ID`` env var.
        commit_message: Commit message shown in the HF repo history.

    Returns:
        The URL of the uploaded folder on HF Hub.
    """
    api = load_hf_api()
    repo_id = repo_id or os.getenv("HF_REPO_ID")
    if not repo_id:
        raise ValueError(
            "No HF repo id: pass repo_id or set the HF_REPO_ID environment variable."
        )
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    url = api.upload_folder(
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"Model pushed to: {url}")
    return url


def download_checkpoint_from_hf(checkpoint_dir, local_dir, pathes=None, repo_id=None):
    os.makedirs(local_dir, exist_ok=True)

    if not repo_id:
        repo_id = os.getenv("HF_REPO_ID")
        if not repo_id:
            raise ValueError("repo_id must be provided or set via HF_REPO_ID environment variable.")

    if not pathes:
        api = load_hf_api()
        # Fetch all files inside the specified subfolder
        repo_files = api.list_repo_tree(
            repo_id=repo_id,
            path_in_repo=checkpoint_dir,
            recursive=True
        )
        # Extract the relative file paths within that folder
        # e.g., converts 'checkpoint-100/pytorch_model.bin' to 'pytorch_model.bin'
        pathes = [
            os.path.relpath(f.path, checkpoint_dir)
            for f in repo_files if not f.path.endswith('/')
        ]

    print(f"Downloading checkpoint from {repo_id}/{checkpoint_dir} to {local_dir}...")

    for filename in pathes:
        # Construct the full remote path inside the repo
        remote_file_path = os.path.join(checkpoint_dir, filename).replace("\\", "/")

        hf_hub_download(
            repo_id=repo_id,
            filename=remote_file_path,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Recommended to keep standard file structures
        )
        print(f"Downloaded {filename}")

    return os.path.join(local_dir, checkpoint_dir)