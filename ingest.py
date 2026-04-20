"""CLI script to ingest markdown files from the docs/ directory."""

import sys
from pathlib import Path

from app.db import index_document, init_db


def ingest_directory(docs_dir: str = "docs"):
    """Read all .md files in a directory and index them."""
    init_db()
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        print(f"Directory '{docs_dir}' not found.")
        sys.exit(1)

    md_files = sorted(docs_path.glob("*.md"))
    if not md_files:
        print(f"No .md files found in '{docs_dir}'.")
        sys.exit(1)

    for f in md_files:
        doc_id = f.stem
        text = f.read_text()
        print(f"Indexing: {doc_id} ({len(text)} chars)")
        index_document(doc_id, text)

    print(f"\nDone. Indexed {len(md_files)} documents.")


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "docs"
    ingest_directory(directory)
