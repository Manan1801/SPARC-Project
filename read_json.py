#!/usr/bin/env python3
"""
Script: read_json.py
Description:
  Loads a single JSON file (path given as argument) and prints its contents.
Usage:
  python read_json.py /path/to/somefile.json
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Load and inspect a single JSON file.")
    parser.add_argument("json_path", type=str, help="Path to the JSON file.")
    args = parser.parse_args()

    json_file = Path(args.json_path)
    if not json_file.is_file():
        print(f"ERROR: JSON file does not exist => {json_file}")
        sys.exit(1)

    # Load the JSON
    with open(json_file, "r") as f:
        data = json.load(f)

    # Print top-level type/structure
    print(f"\n[INFO] Successfully loaded: {json_file}")
    print(f"Type of 'data': {type(data)}")

    # If it's a dict, print keys:
    if isinstance(data, dict):
        print(f"Top-level keys => {list(data.keys())}")
    elif isinstance(data, list):
        print(f"JSON is a list of length => {len(data)}")

    print("\n--- JSON CONTENTS ---")
    # Pretty-print entire JSON data
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
