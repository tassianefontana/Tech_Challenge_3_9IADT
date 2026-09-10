"""Download do dataset PubMedQA (subset PQA-L, rotulado por especialistas)."""

import sys
import urllib.request

from src.config import PUBMEDQA_RAW_FILE, PUBMEDQA_RAW_URL


def download(force: bool = False) -> None:
    if PUBMEDQA_RAW_FILE.exists() and not force:
        size_mb = PUBMEDQA_RAW_FILE.stat().st_size / 1_048_576
        print(f"[skip] {PUBMEDQA_RAW_FILE.name} ja existe ({size_mb:.1f} MB).")
        return

    print(f"[download] {PUBMEDQA_RAW_URL}")
    urllib.request.urlretrieve(PUBMEDQA_RAW_URL, PUBMEDQA_RAW_FILE)
    size_mb = PUBMEDQA_RAW_FILE.stat().st_size / 1_048_576
    print(f"[ok] salvo em {PUBMEDQA_RAW_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    download(force="--force" in sys.argv)
