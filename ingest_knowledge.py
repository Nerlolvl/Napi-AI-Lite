from __future__ import annotations

import argparse
import sys
from pathlib import Path

from storage.db_manager import NapiBrain


DEFAULT_DB_PATH = "./storage/napi_brain.db"
DEFAULT_KNOWLEDGE_DIR = "./knowledge"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Load markdown/txt knowledge into Napi brain.")
    parser.add_argument("directory", nargs="?", default=DEFAULT_KNOWLEDGE_DIR)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"Knowledge directory not found: {directory}")
        return 1

    brain = NapiBrain(args.db)
    chunks = brain.ingest_directory(str(directory))
    print(f"Loaded knowledge chunks: {chunks}")
    print(f"Database: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
