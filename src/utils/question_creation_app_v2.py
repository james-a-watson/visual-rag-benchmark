import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import base64
import matplotlib.pyplot as plt
from PIL import Image
from agents import QuestionGenerationAgent, ImageQuestionCreationAgent


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def display_image(image_path):
    """Display an image using matplotlib with auto-close on Enter"""
    try:
        img = Image.open(image_path)
        fig = plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Image: {os.path.basename(image_path)}")
        plt.tight_layout()

        # Show the plot in non-blocking mode
        plt.show(block=False)
        plt.draw()

        return fig

    except Exception as e:
        print(f"Error displaying image {image_path}: {e}")


def display_text(text_content, max_chars=1000):
    """Display text content with optional truncation"""
    print("=" * 50)
    print("TEXT CONTENT:")
    print("=" * 50)

    if len(text_content) > max_chars:
        print(text_content[:max_chars])
        print(
            f"\n... [Truncated - showing first {max_chars} characters out of {len(text_content)} total]"
        )
    else:
        print(text_content)

    print("=" * 50)


def main(question_mode="both"):
    """
    Main function to create QA pairs with optional content display

    Args:
        question_mode (str): "both", "text", or "table"
        display_content (bool): Whether to display text/images when creating QA pairs
        max_text_chars (int): Maximum characters to display for text content
    """
    QUESTION_SET_PATH = "larger_question_set.json"
    PAGE_DATA_PATH = "page_extraction"

    image_question_creation_agent = ImageQuestionCreationAgent("gpt4o")
    question_creation_agent = QuestionGenerationAgent("gpt-4.1-test")

    new_questions = []
    with open(QUESTION_SET_PATH, "r") as f:
        question_set = json.load(f)["question_set"]

    with open("questions_skip.json", "r") as f:
        questions_skip = json.load(f)

    question_set.extend(questions_skip)

    for pdf_file in os.listdir(PAGE_DATA_PATH):
        pdf_path = os.path.join(PAGE_DATA_PATH, pdf_file)
        for page in os.listdir(pdf_path):
            textqa_exists = False
            tableqa_exists = False

            if question_mode == "text":
                tableqa_exists = True

            if question_mode == "table":
                textqa_exists = True

            page_number = int(page.split("_")[1])
            page_path = os.path.join(pdf_path, page)

            for question in question_set:
                if (
                    question["ground_truth_source"]["document"] == pdf_file
                    and question["ground_truth_source"]["page"] == page_number
                ):
                    if question["ground_truth_source"]["type"] == "text":
                        textqa_exists = True
                    elif question["ground_truth_source"]["type"] == "table":
                        tableqa_exists = True
                    else:
                        pass

            # Handle Text QA
            if not textqa_exists:
                text_file_path = os.path.join(page_path, "text.txt")
                with open(text_file_path, "r") as f:
                    page_text = f.read()

                if len(page_text) > 20:
                    print("=" * 50)
                    print(
                        f"Creating TextQA pair for {pdf_file[14:28]} - Page {page_number}"
                    )
                    response = question_creation_agent.ask(page_text)
                    response_dict = json.loads(response.content)
                    question = response_dict["question"]
                    answer = response_dict["answer"]

                    qa_example = {
                        "question": question,
                        "answer": answer,
                        "ground_truth_source": {
                            "type": "text",
                            "document": pdf_path.split("/")[1],
                            "page": page_number,
                        },
                    }
                    new_questions.append(qa_example)
                    with open(
                        "questions_temp.json", "w", encoding="utf-8"
                    ) as json_file:
                        json.dump(
                            new_questions,
                            json_file,
                            indent=4,
                            ensure_ascii=False,
                        )
            else:
                print("Text QA already exists")

            # Handle Table/Figure QA
            if not tableqa_exists:
                figures_path = os.path.join(page_path, "figures")
                for figure in os.listdir(figures_path):
                    if figure.endswith(".jpg"):
                        print("=" * 50)
                        print(
                            f"\nCreating TableQA pair for {pdf_file[14:28]} - Page {page_number} - {figure}"
                        )
                        figure_path = os.path.join(figures_path, figure)
                        base64_image = encode_image_to_base64(figure_path)
                        response = image_question_creation_agent.ask(base64_image)
                        try:
                            response_dict = json.loads(response)
                        except Exception as e:
                            print("ERRRORRRR")
                            print(response)
                            print("ERRRORRRR")
                        question = response_dict["question"]
                        answer = response_dict["answer"]

                        qa_example = {
                            "question": question,
                            "answer": answer,
                            "ground_truth_source": {
                                "type": "table",
                                "document": pdf_path.split("/")[1],
                                "page": page_number,
                            },
                        }
                        new_questions.append(qa_example)
                        with open(
                            "questions_temp.json", "w", encoding="utf-8"
                        ) as json_file:
                            json.dump(
                                new_questions,
                                json_file,
                                indent=4,
                                ensure_ascii=False,
                            )
            else:
                print("Table QA already exists")


# Example usage:
if __name__ == "__main__":
    # Use this for full functionality with image display
    main(question_mode="text", display_content=True, max_text_chars=500)
