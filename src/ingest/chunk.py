"""Chunking strategies for the two corpus types.

Markdown docs: split on headers, keeping the header path as context (so a chunk
under "## Path Parameters" carries that heading even if the split happens mid-section).

Python source: split on top-level function/class boundaries using `ast`, so a chunk
is always a complete, syntactically valid unit instead of an arbitrary character window.
"""
import ast
import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 2000


@dataclass
class Chunk:
    text: str
    source: str
    kind: str  # "doc" or "code"
    heading_path: str = ""


def chunk_markdown_file(path: Path, root: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(root))
    lines = text.split("\n")

    chunks: list[Chunk] = []
    heading_stack: list[str] = []
    buffer: list[str] = []

    def flush():
        content = "\n".join(buffer).strip()
        if content:
            heading_path = " > ".join(heading_stack) if heading_stack else rel
            for piece in _split_long(content, MAX_CHARS):
                chunks.append(Chunk(text=piece, source=rel, kind="doc", heading_path=heading_path))
        buffer.clear()

    header_re = re.compile(r"^(#{1,6})\s+(.*)")
    for line in lines:
        m = header_re.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            heading_stack = heading_stack[: level - 1] + [title]
        buffer.append(line)
    flush()
    return chunks


def chunk_python_file(path: Path, root: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(root))
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.split("\n")
    chunks: list[Chunk] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", start + 1)
            snippet = "\n".join(lines[start:end]).strip()
            if not snippet:
                continue
            for piece in _split_long(snippet, MAX_CHARS):
                chunks.append(Chunk(text=piece, source=rel, kind="code", heading_path=node.name))
    return chunks


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def chunk_corpus(docs_dir: Path, code_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for md_path in docs_dir.rglob("*.md"):
        chunks.extend(chunk_markdown_file(md_path, docs_dir))
    for py_path in code_dir.rglob("*.py"):
        chunks.extend(chunk_python_file(py_path, code_dir))
    return chunks


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2] / "data" / "raw"
    result = chunk_corpus(root / "docs", root / "code")
    doc_n = sum(1 for c in result if c.kind == "doc")
    code_n = sum(1 for c in result if c.kind == "code")
    print(f"Chunked corpus: {doc_n} doc chunks, {code_n} code chunks, {len(result)} total")
