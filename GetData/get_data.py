import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.request

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def fetch_chembl(output_file: str, source: dict) -> int:
    target_name = source.get("target_name")
    if not target_name:
        logging.error("Missing source.target_name for data_source=chembl")
        return 1

    script_path = os.path.join("GetData", "get_ChEMBL_target_full.py")
    result = subprocess.run(
        [sys.executable, script_path, target_name, output_file],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logging.error("ChEMBL fetch failed. Please retry later or use local cached data.")
        return result.returncode
    return 0


def fetch_local_csv(output_file: str, source: dict) -> int:
    input_path = source.get("path")
    if not input_path:
        logging.error("Missing source.path for data_source=local_csv")
        return 1

    if not os.path.exists(input_path):
        logging.error(f"Local CSV not found: {input_path}")
        return 1

    shutil.copyfile(input_path, output_file)
    return 0


def fetch_http_csv(output_file: str, source: dict) -> int:
    url = source.get("url")
    if not url:
        logging.error("Missing source.url for data_source=http_csv")
        return 1

    try:
        with urllib.request.urlopen(url) as resp, open(output_file, "wb") as out:
            out.write(resp.read())
    except Exception as exc:
        logging.error(f"Failed to download CSV from {url}: {exc}")
        return 1

    return 0


def get_data(output_file: str, data_source: str, source: dict) -> int:
    if data_source == "chembl":
        return fetch_chembl(output_file, source)
    if data_source == "local_csv":
        return fetch_local_csv(output_file, source)
    if data_source == "http_csv":
        return fetch_http_csv(output_file, source)

    logging.error(f"Unknown data_source: {data_source}")
    return 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Fetch raw data from a configured source.")
    parser.add_argument("output_file", type=str, help="Output CSV file path")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON config file")
    parser.add_argument("--data_source", type=str, default=None, help="Data source name")
    parser.add_argument("--source", type=str, default=None, help="JSON object for source config")

    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        data_source = config.get("data_source")
        source = config.get("source", {})
    else:
        data_source = args.data_source
        source = json.loads(args.source) if args.source else {}

    if not data_source:
        logging.error("data_source is required")
        return 1

    return get_data(args.output_file, data_source, source)


if __name__ == "__main__":
    raise SystemExit(main())
