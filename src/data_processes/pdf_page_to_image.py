import os
from pdf2image import convert_from_path
from PIL import ImageEnhance


def enhance_image(img):
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2)
    img = ImageEnhance.Sharpness(img).enhance(7.5)
    return img


def pdf_to_image(pdf_path, pdf_file, output_path=r"output_images", count=0):
    output_images_paths = []
    images = convert_from_path(pdf_path)
    os.makedirs(os.path.join(output_path, pdf_file), exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(output_path, pdf_file, f"page_{i + count + 1}.jpg")
        _image = enhance_image(image)
        _image.save(path, "JPEG")
        output_images_paths.append(path)
        print(f"Saved page_{i + count + 1}.jpg")
    return output_images_paths


if __name__ == "__main__":
    files = [
        # "(0 Very Good) Environmental standards phase 2_Final_110309.pdf",
        # "(0 Very Good) Guidance No 01 - Economics - WATECO (WG 2.6).pdf",
        # "(0 Very Good) Guidance No 27 - Deriving Environmental Quality Standards - version 2018-1.pdf",
        "(0 Very Good) Guidance No 38 - Technical guidance for EQS for metals (1) (1).pdf"
    ]
    for file in files:
        pdf_to_image(f"raw_data/{file}", file)
