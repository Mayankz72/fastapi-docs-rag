"""Pull answered Q&A discussions from GitHub as the golden-dataset source.

FastAPI routes support questions through GitHub Discussions (category "Questions")
rather than issues, and each answered discussion has a marked accepted answer.
That gives us free, real (question, ground-truth-answer) pairs for eval.
"""
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
GRAPHQL_URL = "https://api.github.com/graphql"
OWNER, REPO = "fastapi", "fastapi"
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "discussions.json"

QUERY = """
query($owner: String!, $repo: String!, $cursor: String) {
  repository(owner: $owner, name: $repo) {
    discussions(first: 50, after: $cursor, categoryId: null) {
      pageInfo { hasNextPage endCursor }
      nodes {
        title
        body
        url
        isAnswered
        answer {
          body
        }
      }
    }
  }
}
"""


def fetch_all(max_pages: int = 20) -> list[dict]:
    if not GITHUB_TOKEN:
        raise SystemExit(
            "GITHUB_TOKEN not set in .env. Create a classic PAT with public_repo "
            "read access (no special scopes needed for public discussions)."
        )
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    cursor = None
    collected: list[dict] = []
    for page in range(max_pages):
        variables = {"owner": OWNER, "repo": REPO, "cursor": cursor}
        resp = requests.post(
            GRAPHQL_URL,
            json={"query": QUERY, "variables": variables},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload:
            raise RuntimeError(payload["errors"])
        data = payload["data"]["repository"]["discussions"]
        for node in data["nodes"]:
            if node["isAnswered"] and node.get("answer"):
                collected.append(
                    {
                        "question": node["title"],
                        "question_body": node["body"] or "",
                        "answer": node["answer"]["body"],
                        "url": node["url"],
                    }
                )
        print(f"Page {page + 1}: {len(collected)} answered Q&A pairs so far")
        if not data["pageInfo"]["hasNextPage"]:
            break
        cursor = data["pageInfo"]["endCursor"]
        time.sleep(0.5)
    return collected


if __name__ == "__main__":
    pairs = fetch_all()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"Saved {len(pairs)} answered Q&A pairs -> {OUT_PATH}")
