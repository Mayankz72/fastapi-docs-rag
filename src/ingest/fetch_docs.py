"""Shallow-clone the FastAPI repo and stage the docs + source files we'll index."""
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/tiangolo/fastapi.git"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CLONE_DIR = RAW_DIR / "fastapi_repo"

DOCS_SRC = CLONE_DIR / "docs" / "en" / "docs"
CODE_SRC = CLONE_DIR / "fastapi"

DOCS_DEST = RAW_DIR / "docs"
CODE_DEST = RAW_DIR / "code"


def clone_repo() -> None:
    if CLONE_DIR.exists():
        print(f"Repo already cloned at {CLONE_DIR}, skipping clone.")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)],
        check=True,
    )


def stage_docs() -> None:
    if DOCS_DEST.exists():
        shutil.rmtree(DOCS_DEST)
    shutil.copytree(DOCS_SRC, DOCS_DEST)
    md_count = len(list(DOCS_DEST.rglob("*.md")))
    print(f"Staged {md_count} markdown docs -> {DOCS_DEST}")


def stage_code() -> None:
    if CODE_DEST.exists():
        shutil.rmtree(CODE_DEST)
    shutil.copytree(CODE_SRC, CODE_DEST)
    py_count = len(list(CODE_DEST.rglob("*.py")))
    print(f"Staged {py_count} python source files -> {CODE_DEST}")


if __name__ == "__main__":
    clone_repo()
    stage_docs()
    stage_code()
