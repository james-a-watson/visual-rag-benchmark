import os
import base64
from time import sleep
from agents import ImageDescriptionAgent

ROOT_IMAGE_DIR = "page_extraction"


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_description_agent = ImageDescriptionAgent("gpt4o")
pdf_files = [file for file in os.listdir(ROOT_IMAGE_DIR)]

for file in pdf_files:
    file_path = os.path.join(ROOT_IMAGE_DIR, file)
    for page in os.listdir(file_path):
        figures_path = os.path.join(file_path, page, "figures")
        figure_paths = [
            os.path.join(figures_path, f)
            for f in os.listdir(figures_path)
            if f.endswith(".jpg")
        ]
        description_paths = [
            os.path.join(figures_path, f)
            for f in os.listdir(figures_path)
            if f.endswith("_description.txt")
        ]
        for figure_path in figure_paths:
            description_path = figure_path + "_description.txt"
            if description_path in description_paths:
                print("Description Exists:", description_path)
                continue

            base64_image = encode_image_to_base64(figure_path)
            image_description = image_description_agent.ask(base64_image)
            with open(description_path, "w") as f:
                f.write(image_description)
            print("Description Saved:", description_path)

        if len(figure_paths) > 1:
            sleep(3)
