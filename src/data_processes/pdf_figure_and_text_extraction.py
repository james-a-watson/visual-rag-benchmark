import base64
import os
import shutil

from agents import ImageTextExtractionAgent
from pdf_open_cv import PDFContentAnalyzer
import logging

logging.basicConfig(
    level=logging.INFO,
    format="Page Image Extraction - %(levelname)s - %(message)s",
)

logger = logging.getLogger("Page Image Extraction ")


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    root_image_dir = "output_images"
    output_dir = "page_extraction"

    image_processing_agent = ImageTextExtractionAgent("gpt4o")
    # pdf_name = "(0 Very Good) Environmental standards phase 2_Final_110309.pdf"
    # pdf_name = "(0 Very Good) Guidance No 01 - Economics - WATECO (WG 2.6).pdf"
    pdf_name = "(0 Very Good) Guidance No 27 - Deriving Environmental Quality Standards - version 2018-1.pdf"
    logger.info(f"Extracting {pdf_name}")
    image_dir = os.path.join(root_image_dir, pdf_name)
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        page, _ = image_file.split(".")
        if page in os.listdir(os.path.join(output_dir, pdf_name)):
            logger.info(f"{page} already processed.")
            continue

        logger.info(f"Processing {page}")
        image = encode_image_to_base64(image_path)
        text_output = image_processing_agent.ask(image)

        analyzer = PDFContentAnalyzer(image_path)
        regions = analyzer.classify_regions()

        output_path = os.path.join(output_dir, pdf_name, page)
        os.makedirs(output_path, exist_ok=True)

        shutil.copyfile(image_path, os.path.join(output_path, "original_page.jpg"))
        analyzer.save_cropped_regions(regions, output_path)
        with open(os.path.join(output_path, "text.txt"), "w") as f:
            f.write(text_output)
