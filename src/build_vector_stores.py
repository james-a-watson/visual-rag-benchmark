import logging
import os
import pymupdf4llm
import base64

from uuid import uuid4
from pathlib import Path
from typing import List
from time import time, sleep, perf_counter
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="Vector Store Pipelines - %(levelname)s - %(message)s"
)

logger = logging.getLogger("Vector Store Pipelines")


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class VectorStorePipeline:

    def __init__(
        self,
        root_dir="raw_data",
        pipeline_type: str = "base",
        vector_store_prefix: str = "",
        existing_vector_store: str = "",
    ):
        self.root_dir = root_dir
        self.pipeline_type = pipeline_type
        if existing_vector_store:
            self.vector_store_name = existing_vector_store
            logger.info("Loading from existing vector store")
        else:
            if vector_store_prefix:
                self.vector_store_name = f"{vector_store_prefix}_{int(time())}"
            else:
                self.vector_store_name = f"{self.pipeline_type}_{int(time())}"

        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_NAME"]
        )
        self.vector_store = Chroma(
            collection_name=self.vector_store_name,
            embedding_function=self.embeddings,
            persist_directory="vector_stores",
        )
        logger.info(f"Vector Store Ready: {self.vector_store_name}...")

    def get_top_pdf_files(self, folder_path: str, num_files: int) -> List[Path]:
        """Select top N PDF files from folder, sorted by name."""
        folder = Path(folder_path)
        pdf_files = sorted(folder.glob("*.pdf"))
        return pdf_files[:num_files]

    def prepare_document_chunks(self) -> List[Document]:
        raise NotImplementedError(
            "Base VectorStorePipeline. Please use a specific Pipeline."
        )

    def add_documents_to_vector_store(
        self, chunks: List[Document], batch_size: int = 100
    ):
        """Create vector store from documents with text chunking."""
        chunks_loaded = 0
        for i in range(0, len(chunks), batch_size):
            if i != 0:
                sleep(1)
            chunk_batch = chunks[i : i + batch_size]
            batch_number = i // batch_size + 1
            uuids = [str(uuid4()) for _ in range(len(chunk_batch))]
            self.vector_store.add_documents(documents=chunk_batch, ids=uuids)
            chunks_loaded += len(chunk_batch)
            logger.info(
                f"Batch {batch_number}: {len(chunk_batch)} chunks loaded ({chunks_loaded}/{len(chunks)})."
            )

    def document_pipeline(
        self,
        num_files: int,
        chunk_size: int = 1000,
        batch_size: int = 100,
    ):
        starting_doc_count = self.vector_store._collection.count()
        collection_name = self.vector_store._collection.name
        logger.info(f"---------- PIPELINE START ----------")
        start_time = perf_counter()
        logger.info(f"Vector Store {collection_name} - Doc Count {starting_doc_count}")

        logger.info(f"Processing top {num_files} PDFs from '{self.root_dir}'...")
        pdf_paths = self.get_top_pdf_files(self.root_dir, num_files)
        logger.info(f"Selected {len(pdf_paths)} PDF files")
        page_count = 0
        for path in pdf_paths:
            try:
                page_count += len(os.listdir(path))
            except NotADirectoryError:
                pdf_pages = PyPDFLoader(str(path))
                page_count += len(pdf_pages.load())

        logger.info(f"Total pages {page_count}")

        document_chunks = self.prepare_document_chunks(pdf_paths)
        logger.info(f"Documents for vector store: {len(document_chunks)}")

        self.add_documents_to_vector_store(document_chunks, batch_size)
        logger.info(f"Document load complete.")
        end_time = perf_counter()
        time_taken = end_time - start_time
        ending_doc_count = self.vector_store._collection.count()
        logger.info(f"----------- PIPELINE END -----------")
        logger.info("")
        logger.info("pipeline details:")
        logger.info("-" * 30)
        logger.info(f"pipeline type         {self.pipeline_type}")
        logger.info(f"file count            {len(pdf_paths)}")
        logger.info(f"total page count      {page_count}")
        logger.info(f"chunks created        {len(document_chunks)}")
        logger.info(f"vector store name     {collection_name}")
        logger.info(f"start document count  {starting_doc_count}")
        logger.info(f"final document count  {ending_doc_count}")
        logger.info(f"time taken            {time_taken}")
        logger.info("")


class VectorStorePipelineSimpleText(VectorStorePipeline):

    def __init__(self, vector_store_prefix="", existing_vector_store=""):
        super().__init__(
            pipeline_type="simple_text",
            vector_store_prefix=vector_store_prefix,
            existing_vector_store=existing_vector_store,
        )

    def prepare_document_chunks(
        self, pdf_paths: List[Path], chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(str(pdf_path), mode="single")
            pages = loader.load()
            documents.extend(pages)
        return text_splitter.split_documents(documents)


class VectorStorePipelinePerPageText(VectorStorePipeline):

    def __init__(self, vector_store_prefix="", existing_vector_store=""):
        super().__init__(
            pipeline_type="per_page_text",
            vector_store_prefix=vector_store_prefix,
            existing_vector_store=existing_vector_store,
        )

    def prepare_document_chunks(self, pdf_paths: List[Path]) -> List[Document]:
        documents = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            for i, page in enumerate(pages):
                page_doc = Document(
                    page_content=page.page_content,
                    metadata={
                        "file_path": str(pdf_path),
                        "page": i + 1,
                    },
                )
                documents.append(page_doc)

        return documents


class VectorStorePipelineSimpleMarkdown(VectorStorePipeline):

    def __init__(self, vector_store_prefix="", existing_vector_store=""):
        super().__init__(
            pipeline_type="simple_markdown",
            vector_store_prefix=vector_store_prefix,
            existing_vector_store=existing_vector_store,
        )

    def prepare_document_chunks(
        self, pdf_paths: List[Path], chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        documents = []
        for pdf_path in pdf_paths:
            markdown_text = pymupdf4llm.to_markdown(pdf_path)
            markdown_chunks = text_splitter.create_documents(texts=[markdown_text])
            documents.extend(markdown_chunks)
        return documents


class VectorStorePipelinePerPageMarkdown(VectorStorePipeline):

    def __init__(self, vector_store_prefix="", existing_vector_store=""):
        super().__init__(
            pipeline_type="per_page_markdown",
            vector_store_prefix=vector_store_prefix,
            existing_vector_store=existing_vector_store,
        )

    def prepare_document_chunks(self, pdf_paths: List[Path]) -> List[Document]:
        documents = []
        for pdf_path in pdf_paths:
            markdown_pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
            for i, page in enumerate(markdown_pages):
                page_doc = Document(
                    page_content=page["text"],
                    metadata={
                        "file_path": str(pdf_path),
                        "page": i + 1,
                    },
                )
                documents.append(page_doc)
        return documents


class VectorStorePipelineMultiModalDescriptions(VectorStorePipeline):

    def __init__(
        self,
        root_dir="page_extraction",
        vector_store_prefix="",
        existing_vector_store="",
    ):
        super().__init__(
            root_dir=root_dir,
            pipeline_type="multi_modal_descriptions",
            vector_store_prefix=vector_store_prefix,
            existing_vector_store=existing_vector_store,
        )

    def create_text_documents(
        self, pdf_path, chunk_size: int = 1000, chunk_overlap: int = 200
    ):
        """
        Extracts, into a list, the text contents of each
        page in a PDF extracted into the root_dir.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        page_texts = []
        for page in os.listdir(pdf_path):
            page_text_path = os.path.join(pdf_path, page, "text.txt")
            with open(page_text_path, "r") as f:
                page_text = f.read()

            if len(page_text) > 20:
                page_texts.append(page_text)

        total_pdf_text = "\n".join(page_texts)

        return text_splitter.create_documents(texts=[total_pdf_text])

    def get_figure_descriptions(self, pdf_path):
        figure_description_tuples = []
        for page in os.listdir(pdf_path):
            figures_path = os.path.join(pdf_path, page, "figures")
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
            for figure_path, description_path in zip(figure_paths, description_paths):
                figure_description_tuples.append((figure_path, description_path))

        return figure_description_tuples

    def create_image_documents(self, figure_tuples: List[tuple]) -> List[Document]:
        image_documents = []
        for figure_path, description_path in figure_tuples:
            with open(description_path, "r", encoding="utf-8") as f:
                description_text = f.read().strip()

            image_documents.append(
                Document(
                    page_content=description_text,
                    metadata={
                        "type": "image",
                        "description": description_text,
                        "image_content": encode_image_to_base64(figure_path),
                        "image_path": figure_path,
                        "description_path": description_path,
                    },
                )
            )
        return image_documents

    def prepare_document_chunks(self, pdf_files):
        all_text_documents = []
        all_image_documents = []
        for pdf_path in pdf_files:
            text_documents = self.create_text_documents(pdf_path)
            all_text_documents.extend(text_documents)

            figure_tuples = self.get_figure_descriptions(pdf_path)
            image_documents = self.create_image_documents(figure_tuples)
            all_image_documents.extend(image_documents)

        logger.info(f"Total text documents: {len(all_text_documents)}")
        logger.info(f"Total image documents: {len(all_image_documents)}")
        input("Continue?")
        return all_text_documents + all_image_documents


if __name__ == "__main__":
    vector_store_pipelines = [
        # VectorStorePipelineSimpleText(),
        # VectorStorePipelinePerPageText(),
        # VectorStorePipelineSimpleMarkdown(),
        # VectorStorePipelinePerPageMarkdown(),
        VectorStorePipelineMultiModalDescriptions(),
    ]

    for pipeline in vector_store_pipelines:
        pipeline.document_pipeline(num_files=3)

        # Chunk Size is used for per page here. Could try legit per page thing.
        # simple_text_1752332748
        # per_page_text_1752332748
        # simple_markdown_1752333450
        # simple_markdown_1752333450
