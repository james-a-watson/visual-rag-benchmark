import requests
import os
from time import perf_counter
from PIL import Image
import json
import pickle

from dotenv import load_dotenv

load_dotenv()

# Updated for ngrok URL format
COPALI_API_URL = os.getenv("COPALI_API_URL")
COPALI_API_KEY = os.getenv("COPALI_API_KEY")

headers = {"Authorization": f"Bearer {COPALI_API_KEY}"}


def test_connection():
    """Test if the API is accessible"""
    try:
        response = requests.get(f"{COPALI_API_URL}/health", headers=headers)
        if response.status_code == 200:
            print("✅ API connection successful!")
            print(f"API Status: {response.json()}")
            return True
        else:
            print(f"❌ API connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def embed_text(query_text):
    """Embed a text query"""
    try:
        response = requests.post(
            f"{COPALI_API_URL}/embed_text",
            headers=headers,
            params={"query": query_text},
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Text embedding failed: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"❌ Text embedding error: {e}")
        return None


def get_image_paths(num_pdf_files, num_img_files, root_dir="output_images"):
    """Get image paths from directory structure"""
    img_files_to_process = []
    if not os.path.exists(root_dir):
        print(f"❌ Directory {root_dir} not found")
        return img_files_to_process

    pdf_files = os.listdir(root_dir)
    for pdf_file in pdf_files[:num_pdf_files]:
        pdf_path = os.path.join(root_dir, pdf_file)
        if os.path.isdir(pdf_path):
            img_files = os.listdir(pdf_path)
            img_file_paths = [
                os.path.join(root_dir, pdf_file, img_file)
                for img_file in img_files[:num_img_files]
                if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff"))
            ]
            img_files_to_process.extend(img_file_paths)
    return img_files_to_process


def get_image_size(image_path):
    """Print image dimensions and mode"""
    try:
        img = Image.open(image_path)
        print(f"{image_path}: {img.size} ({img.mode})")
        return img.size
    except Exception as e:
        print(f"❌ Error reading {image_path}: {e}")
        return None


def resize_image(image_path, max_width=600):
    """Resize image (saved for future use)"""
    img = Image.open(image_path)
    ratio = max_width / img.width
    new_height = int(img.height * ratio)
    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return img


def embed_single_image(image_path):
    """Embed a single image"""
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}

            start_time = perf_counter()
            response = requests.post(
                f"{COPALI_API_URL}/embed_image",
                headers=headers,
                files=files,
            )
            end_time = perf_counter()

        if response.status_code == 200:
            result = response.json()
            print(
                f"✅ Single image embedding completed in {end_time - start_time:.2f}s"
            )
            return result
        else:
            print(f"❌ Single image embedding failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Single image embedding error: {e}")
        return None


def embed_images_batch(image_paths):
    """Embed multiple images in a single batch request"""
    try:
        open_files = []
        file_handles = []

        # Prepare files for upload
        for image_path in image_paths:
            file_handle = open(image_path, "rb")
            file_handles.append(file_handle)
            open_files.append(
                ("images", (os.path.basename(image_path), file_handle, "image/jpeg"))
            )

        start_time = perf_counter()
        response = requests.post(
            f"{COPALI_API_URL}/embed_images_batch",
            headers=headers,
            files=open_files,
        )
        end_time = perf_counter()

        # Close file handles
        for file_handle in file_handles:
            file_handle.close()

        print(f"⏱️  Time taken for batch API: {end_time - start_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch embedding completed!")
            print(f"📊 Total Images: {result['total_images']}")
            print(f"⏱️  Server Time: {result['inference_time']}")
            print(f"📈 Avg Time per Image: {result['avg_time_per_image']}")
            return result
        else:
            print(f"❌ Batch embedding failed: {response.status_code}")
            if response.status_code == 422:
                print("Validation Error:", response.json())
            else:
                print(response.text)
            return None

    except Exception as e:
        print(f"❌ Batch embedding error: {e}")
        return None


def search_documents(image_paths, queries):
    """Search through documents with text queries"""
    try:
        open_files = []
        file_handles = []

        # Prepare files for upload
        for image_path in image_paths:
            file_handle = open(image_path, "rb")
            file_handles.append(file_handle)
            open_files.append(
                ("images", (os.path.basename(image_path), file_handle, "image/jpeg"))
            )

        # Prepare query data
        data = {}
        for i, query in enumerate(queries):
            data[f"queries"] = query  # FastAPI will handle list automatically

        start_time = perf_counter()
        response = requests.post(
            f"{COPALI_API_URL}/search",
            headers=headers,
            files=open_files,
            data=(
                {"queries": queries}
                if isinstance(queries, list)
                else {"queries": [queries]}
            ),
        )
        end_time = perf_counter()

        # Close file handles
        for file_handle in file_handles:
            file_handle.close()

        print(f"⏱️  Search completed in {end_time - start_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document search completed!")
            return result
        else:
            print(f"❌ Document search failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Document search error: {e}")
        return None


def single_query(image_path, query_text):
    """Simple single image, single query"""
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"query": query_text}

            start_time = perf_counter()
            response = requests.post(
                f"{COPALI_API_URL}/single_query",
                headers=headers,
                files=files,
                data=data,
            )
            end_time = perf_counter()

        print(f"⏱️  Single query completed in {end_time - start_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query: '{query_text}'")
            print(f"📊 Similarity Score: {result['similarity_score']:.4f}")
            return result
        else:
            print(f"❌ Single query failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Single query error: {e}")
        return None


import numpy as np
import json


def convert_to_lists(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_lists(item) for item in obj]
    else:
        return obj


def deep_convert_to_float(obj):
    if isinstance(obj, np.ndarray):
        return obj.astype(float).tolist()
    elif isinstance(obj, list):
        return [deep_convert_to_float(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    else:
        return obj


def similarity_score_from_embeddings(
    query_embeddings_list, image_embeddings_list, batch_size=600
):
    """Calculate similarity score from pre-computed embeddings"""
    try:
        print("Deep convert to float?")
        query_embeddings_list = deep_convert_to_float(query_embeddings_list)
        image_embeddings_list = deep_convert_to_float(image_embeddings_list)
        print("Deep convert done")

        i = 0
        results = []
        while i < len(image_embeddings_list):
            image_embeddings_batch = image_embeddings_list[i : i + batch_size]
            i += batch_size

            # Just do this:
            payload = pickle.dumps(
                {
                    "query_embeddings": query_embeddings_list,
                    "image_embeddings": image_embeddings_batch,
                }
            )
            start_time = perf_counter()
            response = requests.post(
                f"{COPALI_API_URL}/similarity_score",
                headers=headers,
                data=payload,  # requests handles JSON serialization
            )
            end_time = perf_counter()
            print(
                f"⏱️  Similarity calculation completed in {end_time - start_time:.2f}s"
            )

            if response.status_code == 200:
                result = response.json()
                results.append(result)

        if len(results) == len(image_embeddings_list):
            similarity_scores = result["similarity_scores"]
            print(f"📊 Similarity Scores Returned: {len(similarity_scores)}")
            return result
        else:
            print(f"❌ Similarity calculation failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Similarity calculation error: {e}")
        return None


def server_side_similarity_query(query_text):
    """Calculate similarity score from pre-computed embeddings"""
    try:
        print("Starting Server Side request")
        start_time = perf_counter()
        response = requests.post(
            f"{COPALI_API_URL}/server_side_similarity_score",
            headers=headers,
            params={"query": query_text},
        )
        end_time = perf_counter()
        print(f"⏱️  Similarity calculation completed in {end_time - start_time:.2f}s")

        if response.status_code == 200:
            result = response.json()
            similarity_scores = result["similarity_scores"]
            print(f"📊 Similarity Scores Returned: {len(similarity_scores)}")
            return result
        else:
            print(f"❌ Similarity calculation failed: {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Similarity calculation error: {e}")
        return None


# Main execution
if __name__ == "__main__":
    # Test connection first
    if not test_connection():
        print("Please check your API URL and key in the .env file")
        exit()

    # Configuration
    num_pdf_files = 1
    num_img_files = 1

    # Get image paths
    image_paths = get_image_paths(
        num_pdf_files, num_img_files, root_dir="output_images"
    )

    if not image_paths:
        print("❌ No images found. Check your directory structure.")
        exit()

    print(f"📁 Found {len(image_paths)} images:")
    for i, image_path in enumerate(image_paths):
        get_image_size(image_path)

    # Example usage - uncomment what you want to test:

    # Test text embedding
    # text_result = embed_text("What is the main topic of this document?")

    # Test single image embedding
    if image_paths:
        single_result = embed_single_image(image_paths[0])

    # Test batch image embedding
    # batch_result = embed_images_batch(image_paths)

    # Test single query
    # if image_paths:
    #     query_result = single_query(image_paths[0], "What is in this document?")

    # Test document search
    # search_result = search_documents(image_paths, ["What is the main topic?", "Are there any charts?"])

    print("🎉 Client ready! Uncomment the functions you want to test.")
