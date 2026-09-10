"""Download the source corpus described by corpus/manifest.json.

The three Water Framework Directive guidance documents are public EU
publications, but are not redistributed in this repository. Add the download
URL for each entry in the manifest, then run this script to populate
corpus/pdfs/.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "corpus" / "manifest.json"
PDF_DIR = ROOT / "corpus" / "pdfs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-checksums",
        action="store_true",
        help="record the sha256 of each fetched file back into the manifest",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    documents = manifest["documents"]
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    missing_urls = [d["id"] for d in documents if not d.get("url")]
    if missing_urls:
        print(
            "No URL set for: " + ", ".join(missing_urls),
            file=sys.stderr,
        )
        print(
            f"Add the download URLs to {MANIFEST.relative_to(ROOT)} and re-run. "
            "The documents are listed by title in the manifest and are available "
            "from the WFD guidance document library.",
            file=sys.stderr,
        )
        return 1

    changed = False
    for doc in documents:
        dest = PDF_DIR / doc["filename"]

        if dest.exists():
            print(f"present  {doc['filename']}")
        else:
            print(f"fetching {doc['filename']}")
            download(doc["url"], dest)

        actual = sha256(dest)
        expected = doc.get("sha256")

        if args.write_checksums:
            if expected != actual:
                doc["sha256"] = actual
                changed = True
            print(f"         sha256 {actual}")
        elif expected is None:
            print(f"         sha256 {actual} (not recorded in manifest)")
        elif expected != actual:
            print(
                f"         CHECKSUM MISMATCH\n"
                f"           expected {expected}\n"
                f"           actual   {actual}",
                file=sys.stderr,
            )
            return 1

    if changed:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"\nupdated {MANIFEST.relative_to(ROOT)}")

    print(f"\n{len(documents)} documents in {PDF_DIR.relative_to(ROOT)}")
    print("Next: python scripts/extract_pages.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
