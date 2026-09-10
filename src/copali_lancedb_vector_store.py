import logging
import os
import numpy as np
import base64
import time
import json
from typing import List, Dict, Any, Optional
from uuid import uuid4
from time import time
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LanceDB imports
import lancedb
from lancedb.pydantic import LanceModel
from pydantic import BaseModel

# Import your existing functions
from copali_api_interface import (
    embed_single_image,
    similarity_score_from_embeddings,
    embed_text,
    server_side_similarity_query,
)

logging.basicConfig(
    level=logging.INFO, format="ColPali LanceDB - %(levelname)s - %(message)s"
)

logger = logging.getLogger("ColPali LanceDB")


class ImageDocument(LanceModel):
    """Schema for storing image documents with ColPali embeddings"""

    id: str
    image_path: str
    image_name: str
    image_base64: Optional[str] = None
    image_size: Optional[int] = None
    metadata_json: str = "{}"  # Store metadata as JSON string
    embeddings: List[List[float]]  # ColPali embeddings - any shape
    embedding_shape: List[int]
    created_at: float


class TextDocument(LanceModel):
    """Schema for storing text documents with embeddings"""

    id: str
    content: str
    title: Optional[str] = None
    file_path: Optional[str] = None
    chunk_index: Optional[int] = None  # For document chunks
    parent_doc_id: Optional[str] = (
        None  # Reference to parent document if this is a chunk
    )
    metadata_json: str = "{}"  # Store metadata as JSON string
    embeddings: List[List[float]]  # ColPali embeddings - any shape
    embedding_model: str = "default"  # Track which embedding model was used
    created_at: float


class ColPaliLanceDBStore:
    """
    Simple LanceDB vector store that uses your existing ColPali API functions.
    Preserves multi-dimensional embeddings without modification.
    """

    def __init__(
        self,
        collection_name,
        persist_directory: str = "./lancedb",
        table_name: str = "colpali_images",
        store_images_as_base64: bool = True,
    ):
        self.db_path = os.path.join(persist_directory, collection_name)
        self.table_name = table_name
        self.store_images_as_base64 = store_images_as_base64
        self._collection_name = collection_name

        # Initialize LanceDB
        self.db = lancedb.connect(self.db_path)

        # Create or connect to table
        try:
            self.table = self.db.open_table(table_name)
            logger.info(f"✅ Opened existing table: {table_name}")
        except:
            # Create new table
            self.table = self.db.create_table(
                table_name, schema=ImageDocument, mode="overwrite"
            )
            logger.info(f"✅ Created new table: {table_name}")

        try:
            self.text_table = self.db.open_table("texts")
            logger.info(f"✅ Opened existing table: {table_name}")
        except:
            # Create new table
            self.text_table = self.db.create_table(
                "texts", schema=TextDocument, mode="overwrite"
            )
            logger.info(f"✅ Created new table texts")

        document_count = len(self.table.to_pandas())
        text_document_count = len(self.text_table.to_pandas())
        logger.info(
            f"ColPali LanceDB Store Ready - {document_count} images, {text_document_count} texts"
        )

    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def decode_base64_to_bytes(self, base64_string: str) -> bytes:
        """Convert base64 string back to bytes."""
        return base64.b64decode(base64_string)

    def add_images(
        self,
        image_paths: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add images using your existing embedding functions.
        """
        logger.info(f"Adding {len(image_paths)} images...")

        documents_to_add = []

        for i, image_path in enumerate(image_paths):
            try:
                # Use your embedding function
                embedding_result = embed_single_image(image_path)
                if not embedding_result:
                    logger.warning(f"Skipping {image_path} - embedding failed")
                    continue

                embeddings = np.array(embedding_result["embeddings"])

                # Prepare document
                doc_id = ids[i] if ids and i < len(ids) else str(uuid4())
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}

                # Handle image storage
                image_base64 = None
                image_size = None

                if self.store_images_as_base64:
                    try:
                        image_base64 = self.encode_image_to_base64(image_path)
                    except Exception as e:
                        logger.warning(f"Failed to encode {image_path}: {e}")

                try:
                    image_size = (
                        os.path.getsize(image_path)
                        if os.path.exists(image_path)
                        else None
                    )
                except:
                    image_size = None

                # Create document
                document = ImageDocument(
                    id=doc_id,
                    image_path=str(image_path),
                    image_name=Path(image_path).name,
                    image_base64=image_base64,
                    image_size=image_size,
                    metadata_json=json.dumps(metadata),  # Convert to JSON string
                    embeddings=embeddings.tolist(),
                    embedding_shape=list(embeddings.shape),
                    created_at=time(),
                )

                # documents_to_add.append(document)
                self.table.add([document])
                logger.info(
                    f"✅ Added {Path(image_path).name} - shape: {embeddings.shape}"
                )

            except Exception as e:
                logger.error(f"❌ Error processing {image_path}: {e}")
                continue

    def add_texts(
        self,
        text_chunks: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100000,
    ) -> None:
        """
        Add images using your existing embedding functions.
        """
        logger.info(f"Adding {len(text_chunks)} texts...")

        for i, text in enumerate(text_chunks):
            try:
                # Use your embedding function
                embedding_result = embed_text(text)
                embeddings = np.array(embedding_result["embeddings"])

                # Prepare document
                doc_id = ids[i] if ids and i < len(ids) else str(uuid4())
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}

                # Handle image storage
                document = TextDocument(
                    id=doc_id,
                    content=text,
                    metadata_json=json.dumps(metadata),
                    embeddings=embeddings.tolist(),
                    embedding_shape=list(embeddings.shape),
                    created_at=time(),
                )

                self.text_table.add([document])
                logger.info(f"✅ Added text chunk {i}")

            except Exception as e:
                logger.error(f"❌ Error processing text {i}: {e}")
                continue

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        server_size_db: str = False,
    ) -> List[Dict[str, Any]]:
        """
        Query using your existing functions.
        """

        # Get all documents
        df = self.table.to_pandas()

        # Apply metadata filter if provided
        if metadata_filter:
            for key, value in metadata_filter.items():
                # Parse JSON metadata and filter
                mask = df["metadata_json"].apply(
                    lambda x: json.loads(x).get(key) == value if x else False
                )
                df = df[mask]

        if len(df) == 0:
            return []

        # Calculate similarities using your function
        similarities = []
        valid_indices = []
        image_embeddings = []
        for idx, row in df.iterrows():

            stored_embeddings = row["embeddings"]
            if isinstance(stored_embeddings, np.ndarray):
                stored_embeddings = stored_embeddings.tolist()
            elif isinstance(stored_embeddings, list):
                # It's already a list, but might contain numpy arrays
                stored_embeddings = [
                    item.tolist() if isinstance(item, np.ndarray) else item
                    for item in stored_embeddings
                ]
            image_embeddings.append(stored_embeddings)
            valid_indices.append(idx)

        # Use your similarity function
        if server_size_db:
            sim_result = server_side_similarity_query(query_text)
        else:
            # Get query embedding using your function
            query_result = embed_text(query_text)
            if not query_result:
                logger.error("Failed to get query embedding")
                return []

            query_embeddings = query_result["embeddings"]

            sim_result = similarity_score_from_embeddings(
                [query_embeddings], image_embeddings
            )

        if sim_result:
            similarities = sim_result["similarity_scores"]

        if not similarities:
            return []

        if len(valid_indices) != len(similarities):
            raise Exception(
                "Something went wrong. Fewer scores were returned than rows of the df."
            )

        # Get top results
        top_indices = np.argsort(similarities)[-n_results:][::-1]
        print(len(top_indices))
        # Format results
        results = []
        for i in top_indices:
            row_idx = valid_indices[i]
            row = df.iloc[row_idx]

            result = {
                "id": row["id"],
                "image_path": row["image_path"],
                "image_name": row["image_name"],
                "image_base64": row["image_base64"],
                "metadata": json.loads(row["metadata_json"]),  # Convert back to dict
                "score": similarities[i],
            }
            results.append(result)

        return results

    def text_query(
        self,
        query_text: str,
        n_results: int = 5,
        server_size_db: str = False,
    ) -> List[Dict[str, Any]]:
        """
        Query using your existing functions.
        """
        df = self.text_table.to_pandas()

        similarities = []
        valid_indices = []
        text_embeddings = []
        for idx, row in df.iterrows():

            stored_embeddings = row["embeddings"]
            if isinstance(stored_embeddings, np.ndarray):
                stored_embeddings = stored_embeddings.tolist()
            elif isinstance(stored_embeddings, list):
                # It's already a list, but might contain numpy arrays
                stored_embeddings = [
                    item.tolist() if isinstance(item, np.ndarray) else item
                    for item in stored_embeddings
                ]
            text_embeddings.append(stored_embeddings)
            valid_indices.append(idx)

        # Use your similarity function
        if server_size_db:
            sim_result = server_side_similarity_query(query_text)
        else:
            # Get query embedding using your function
            query_result = embed_text(query_text)
            if not query_result:
                logger.error("Failed to get query embedding")
                return []

            query_embeddings = query_result["embeddings"]

            sim_result = similarity_score_from_embeddings(
                [query_embeddings], text_embeddings
            )

        if sim_result:
            similarities = sim_result["similarity_scores"]

        if not similarities:
            return []

        if len(valid_indices) != len(similarities):
            raise Exception(
                "Something went wrong. Fewer scores were returned than rows of the df."
            )

        # Get top results
        top_indices = np.argsort(similarities)[-n_results:][::-1]
        print(len(top_indices))
        # Format results
        results = []
        for i in top_indices:
            row_idx = valid_indices[i]
            row = df.iloc[row_idx]

            result = {
                "id": row["id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),  # Convert back to dict
                "score": similarities[i],
            }
            results.append(result)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        table_names = self.db.table_names()
        table_stats = []
        for table in table_names:
            db_table = self.db.open_table(table)
            table_stats.append(
                {
                    "document_count": db_table.count_rows(),
                    "table_name": table,
                    "db_path": self.db_path,
                }
            )
        return table_stats

    def load_image_from_result(self, result: Dict[str, Any]) -> Optional[bytes]:
        """Load image bytes from a query result."""
        if self.store_images_as_base64 and result.get("image_base64"):
            return self.decode_base64_to_bytes(result["image_base64"])
        elif result.get("image_path") and os.path.exists(result["image_path"]):
            with open(result["image_path"], "rb") as f:
                return f.read()
        return None


def create_per_page_store():
    vector_store_name = f"per_page_copali_{int(time())}"
    store = ColPaliLanceDBStore(
        vector_store_name,
        persist_directory="./colpali_lancedbs/",
        table_name="documents",
    )

    def get_image_paths(num_pdf_files, num_img_files, root_dir="output_images"):
        img_files_to_process = []
        pdf_files = os.listdir(root_dir)
        for pdf_file in pdf_files[:num_pdf_files]:
            img_files = os.listdir(os.path.join(root_dir, pdf_file))
            img_file_paths = [
                os.path.join(root_dir, pdf_file, img_file)
                for img_file in img_files[:num_img_files]
            ]
            img_files_to_process.extend(img_file_paths)
        return img_files_to_process

    source_dir = "output_images"
    num_pdf_files = 3
    num_img_files = 1000
    image_paths = get_image_paths(num_pdf_files, num_img_files, root_dir=source_dir)

    metadatas = []
    for image_path in image_paths:
        _, pdf_file, jpg_file = image_path.split("/")
        page = jpg_file.split("_")[1].rstrip(".jpg")
        metadatas.append({"file_path": pdf_file, "page": page})

    store.add_images(image_paths, metadatas)

    print(f"Vector Store: {vector_store_name}")
    print(f"📊 Stats: {store.get_stats()}")


def prepare_text_chunks(text_paths):
    all_texts = []
    for text_path in text_paths:
        with open(text_path, "r") as f:
            page_text = f.read()

        if len(page_text) < 20:
            continue

        all_texts.append(page_text)

    full_texts = "\n\n".join(all_texts)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(full_texts)
    return [Document(page_content=text_chunk) for text_chunk in chunks]


def get_images_and_texts(page_extraction_dir: str = "page_extraction"):
    image_paths = []
    text_paths = []
    for extracted_pdf in os.listdir(page_extraction_dir):
        extracted_pdf_path = os.path.join(page_extraction_dir, extracted_pdf)
        for page in os.listdir(extracted_pdf_path):
            figure_dir = os.path.join(extracted_pdf_path, page, "figures")
            figure_paths = [
                os.path.join(figure_dir, fig)
                for fig in os.listdir(figure_dir)
                if fig.endswith(".jpg")
            ]
            image_paths.extend(figure_paths)
            text_path = os.path.join(extracted_pdf_path, page, "text.txt")
            text_paths.append(text_path)

    return image_paths, text_paths


from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_chunked_vector_store():
    vector_store_name = f"chunked_copali_{int(time())}"
    # image_store = ColPaliLanceDBStore(
    #     vector_store_name,
    #     persist_directory="./colpali_lancedbs/",
    #     table_name="documents",
    # )

    print(f"Vector Store Created: {vector_store_name}")
    image_paths, text_paths = get_images_and_texts()
    print(f"Images Found: {len(image_paths)}")
    print(f"Texts Found: {len(text_paths)}")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"]
    )
    vector_store = Chroma(
        collection_name=vector_store_name,
        embedding_function=embeddings,
        persist_directory="vector_stores",
    )
    text_chunks = prepare_text_chunks(text_paths)

    def batch_process(documents_arr, batch_size, process_function):
        for i in range(0, len(documents_arr), batch_size):
            print(f"Documents processed {i}/{len(documents_arr)}")
            batch = documents_arr[i : i + batch_size]
            process_function(batch)

    def add_to_chroma_database(batch):
        vector_store.add_documents(documents=batch)

    batch_process(text_chunks, 50, add_to_chroma_database)

    # store.add_texts(text_chunks[0:20], batch_size=20)
    # image_store.add_images(image_paths)
    print(f"Vector Store: {vector_store_name}")
    # print(f"📊 Stats: {store.get_stats()}")


if __name__ == "__main__":
    vector_store_name = f"full_{int(time())}"
    # create_per_page_store()
    create_chunked_vector_store()
