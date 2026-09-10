"""Render each page of the corpus PDFs to an image for the vision-based methods.

Page images are derived artefacts rather than committed files. The contrast and
sharpness enhancement applied here matches the preprocessing used to produce the
published results, so regenerating the images reproduces the reported figures.

Requires poppler (`brew install poppler`, or `apt install poppler-utils`).
"""

import argparse
import json
from pathlib import Path

from pdf2image import convert_from_path
from PIL import ImageEnhance

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "corpus" / "manifest.json"
PDF_DIR = ROOT / "corpus" / "pdfs"
IMAGE_DIR = ROOT / "corpus" / "page_images"


def enhance_image(img):
    """Greyscale, then raise contrast and sharpness so table rules survive."""
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2)
    img = ImageEnhance.Sharpness(img).enhance(7.5)
    return img


def extract(pdf_path: Path, out_dir: Path, dpi: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = convert_from_path(str(pdf_path), dpi=dpi)
    for i, image in enumerate(images, start=1):
        enhance_image(image).save(out_dir / f"page_{i}.jpg", "JPEG")
    return len(images)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=200, help="render resolution")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-render documents whose page images already exist",
    )
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    total = 0

    for doc in manifest["documents"]:
        pdf_path = PDF_DIR / doc["filename"]
        if not pdf_path.exists():
            print(f"missing  {doc['filename']} — run scripts/fetch_corpus.py first")
            continue

        out_dir = IMAGE_DIR / doc["id"]
        if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
            count = len(list(out_dir.glob("page_*.jpg")))
            print(f"present  {doc['id']}: {count} pages")
            total += count
            continue

        print(f"rendering {doc['id']}")
        count = extract(pdf_path, out_dir, args.dpi)
        print(f"          {count} pages -> {out_dir.relative_to(ROOT)}")
        total += count

    print(f"\n{total} page images in {IMAGE_DIR.relative_to(ROOT)}")
    print("Next: python src/build_vector_stores.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
