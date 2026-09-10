import os
import base64
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from copali_lancedb_vector_store import ColPaliLanceDBStore

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser
from openai import AzureOpenAI

load_dotenv()

EMBEDDING_MODEL = os.environ["AZURE_OPENAI_EMBEDDING_NAME"]


def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


qa_system_prompt = """
You are a helpful assistant. You will answer the users question 
using the context provided. Please use only the context provided.

The question will always have a short answer. Please answer with 
only the direct 1-5 word answer without rephrasing the question 
or providing extra context. 
"""

chain_of_thought_prompt = """
Q: What is the 25th percentile chemistry input to compare fo Sodium
A: The 25th percentile is on row 3. Sodium has symbol Na so looking 
down the column for Na to row 3 gives the answer 22.4. So the final
answer is 22.4
"""


class SimpleRAGAgent:

    def __init__(self, model, vector_store, chain_of_thought=False):
        self.system_prompt = qa_system_prompt
        self.chain_of_thought = chain_of_thought
        if chain_of_thought:
            self.system_prompt += chain_of_thought_prompt
        self.llm = AzureChatOpenAI(model=model)
        self.embeddings = AzureOpenAIEmbeddings(azure_deployment=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory="vector_stores",
            collection_name=vector_store,
            embedding_function=self.embeddings,
        )
        self.prompt = ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("user", "{question} Context: {context}"),
            ]
        )
        self.chain = self.prompt | self.llm

    def ask(self, question, return_docs=False):
        docs = self.vector_store.similarity_search_with_score(question)
        context = "\n\n".join([doc[0].page_content for doc in docs])
        inputs = {"question": question, "context": context}
        if return_docs:
            return self.chain.invoke(inputs).content, docs
        return self.chain.invoke(inputs).content


class QuestionGenerationAgent:

    def __init__(self, model):
        system_prompt = """
        You are an export Quality Assurance Analyst with 20 years experience.
        Your task is to write a question to evaluate a RAG model based on the
        text provided. You are not trying to trick the model so the question
        should be very clear. The answer to each question should be 1-5 words
        and not include anything other than the succinct information.

        Make your questions something that can only be answered by this section
        of text. Your question should NEVER be about parts of the document such 
        as sections, appendices or paragraphs. Imagine the model has access to 
        many such pages and documents so the question should be only about the 
        content that is answerable in the context of a large corpus.

        Please just respond with the following output format.
        {{
        "question": "<your question>",
        "answer": "<1-5 word answer>"
        }}
        """
        self.llm = AzureChatOpenAI(model=model)
        self.prompt = ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("user", "Text: {text}"),
            ]
        )
        self.chain = self.prompt | self.llm

    def ask(self, text):
        return self.chain.invoke({"text": text})


class ValidationResult(BaseModel):
    correct_answer: Literal["Y", "N"] = Field(
        description="Is the answer correct? (Y or N)"
    )
    similarity_score: int = Field(
        description="The similarity of the answers from 1 to 5",
        ge=1,  # greater than or equal to 1
        le=5,  # less than or equal to 5
    )


class AnswerCheckerAgent:

    def __init__(self, model):
        system_prompt = """
        You are a response checker and your task is to look at two
        answers to a question. One of the answers is predefined and
        correct, the other is an LLMs answer. 

        Respond with a Y or N for the correctness of the answer and 
        a similarity score 1-5. 1 means the answers are completely 
        different and 5 means they are exactly the same in meaning. 
        """

        input_prompt = """
        Judge the following 
        Question: {question}
        Correct Answer: {true_answer}
        LLM Answer: {llm_answer}
        {format_instructions}
        """

        self.llm = AzureChatOpenAI(model=model)
        self.parser = PydanticOutputParser(pydantic_object=ValidationResult)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", input_prompt),
            ]
        )

        self.chain = self.prompt | self.llm | self.parser

    def check(self, question, true_answer, llm_answer):
        result = self.chain.invoke(
            {
                "question": question,
                "true_answer": true_answer,
                "llm_answer": llm_answer,
                "format_instructions": self.parser.get_format_instructions(),
            }
        )
        return result


class ImageQuestionCreationAgent:

    def __init__(self, model):
        self.llm_client = AzureOpenAI()
        self.model = model
        self.system_prompt = """
        You are an expert Quality Assurance Analyst with 20 years experience.
        Your task is to write a question to evaluate a RAG model based on the
        table image provided. You are not trying to trick the model so the question
        should be very clear. The answer to each question should be 1-5 words
        and not include anything other than the succinct information.

        Make your questions something that can only be answered by this table. 
        Your question should NEVER be about parts of the document such 
        as sections, appendices or paragraphs. Imagine the model has access to 
        many such pages and documents so the question should be only about the 
        content that is answerable in the context of a large corpus.

        Please just respond with the following output format.
        {{
        "question": "<your question>",
        "answer": "<1-5 word answer>"
        }}
        """

    def ask(self, base64_image):
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        }
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=100,
        )
        return response.choices[0].message.content


class ImageTextExtractionAgent:

    def __init__(self, model):
        self.llm_client = AzureOpenAI()
        self.system_prompt = """
        You are an image parser. Your task is to take in an image of 
        a PDF page and transcribe the text. Your output should only 
        include the text and never any tables or diagrams. Only the 
        paragraphs of text.  

        If there is no text to parse. Just respond with "No Text"
        """

    def ask(self, base64_image):
        response = self.llm_client.chat.completions.create(
            model="gpt4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        }
                    ],
                },
            ],
            temperature=0.0,
            max_tokens=100,
        )
        return response.choices[0].message.content


class ImageDescriptionAgent:

    def __init__(self, model):
        self.llm_client = AzureOpenAI()
        self.system_prompt = """
        You are a specialist in table description. You will 
        be given an image of a table or diagram and your task
        is to return a one-sentence description of that image. 

        The description should be a clear description of what 
        the table contains without mentioning explicit values. 

        Don't mention the table or the image, just the information. 
        
        Example: Metal sulfides along with their logarithmic 
        equilibrium constants from two different sources.
        """

    def ask(self, base64_image):
        response = self.llm_client.chat.completions.create(
            model="gpt4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        }
                    ],
                },
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content


class ImageDescriptionRAGAgent:

    def __init__(self, model, vector_store, chain_of_thought=False):
        self.llm_client = AzureOpenAI()
        self.system_prompt = qa_system_prompt
        self.chain_of_thought = chain_of_thought
        self.messages = [{"role": "system", "content": self.system_prompt}]
        if chain_of_thought:
            image = encode_image_to_base64(
                "output_images/(0 Very Good) Guidance No 38 - Technical guidance for EQS for metals (1) (1).pdf/page_42.jpg"
            )
            self.messages += [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the 25th percentile chemistry input to compare fo Sodium",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": """For this question the information from the table is required. 
                            The 25th percentile is on row 3. Sodium has symbol Na so looking 
                            down the column for Na to row 3 gives the answer 22.4. So the final
                            answer is 22.4""",
                        },
                        {
                            "type": "text",
                            "text": "22.4",
                        },
                    ],
                },
            ]

        self.llm = AzureChatOpenAI(model=model)
        self.embeddings = AzureOpenAIEmbeddings(azure_deployment=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory="vector_stores",
            collection_name=vector_store,
            embedding_function=self.embeddings,
        )

    def ask(self, question, return_docs=False):
        docs = self.vector_store.similarity_search_with_score(question)
        content = [{"type": "text", "text": question}]
        documents_info = []
        for doc, score in docs:
            if "image_content" in doc.metadata:
                base64_image = doc.metadata["image_content"]
                _, file_path, page, _, figure = doc.metadata["image_path"].split("/")
                doc_info = {
                    "file_path": file_path,
                    "page": page.split("_")[1],
                    "figure": figure.split(".")[0],
                    "description": doc.metadata["description"],
                }
                content_block = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high",
                    },
                }
            else:
                doc_info = {"text": doc.page_content}
                content_block = {"type": "text", "text": doc.page_content}
            documents_info.append((doc_info, score))
            content.append(content_block)

        response = self.llm_client.chat.completions.create(
            model="gpt4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
        )
        response_content = response.choices[0].message.content
        if return_docs:
            return response_content, documents_info
        return response_content


class CoPaliRAGAgent:

    def __init__(
        self, model, vector_store, server_size_db=False, chain_of_thought=False
    ):
        self.model = model
        self.server_size_db = server_size_db
        self.system_prompt = qa_system_prompt
        self.chain_of_thought = chain_of_thought
        self.messages = [{"role": "system", "content": self.system_prompt}]
        if chain_of_thought:
            image = encode_image_to_base64(
                "output_images/(0 Very Good) Guidance No 38 - Technical guidance for EQS for metals (1) (1).pdf/page_42.jpg"
            )
            self.messages += [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the 25th percentile chemistry input to compare fo Sodium",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": """For this question the information from the table is required. 
                            The 25th percentile is on row 3. Sodium has symbol Na so looking 
                            down the column for Na to row 3 gives the answer 22.4. So the final
                            answer is 22.4""",
                        },
                        {
                            "type": "text",
                            "text": "22.4",
                        },
                    ],
                },
            ]

        self.llm_client = AzureOpenAI()
        self.vector_store = ColPaliLanceDBStore(
            vector_store,
            persist_directory=f"./colpali_lancedbs",
            table_name="documents",
            store_images_as_base64=True,
        )

    def ask(self, question, return_docs=False):
        docs = self.vector_store.query(
            question, n_results=4, server_size_db=self.server_size_db
        )
        content = [{"type": "text", "text": question}]
        documents_info = []
        for doc in docs:
            if self.server_size_db:
                base64_image = encode_image_to_base64(doc["image_path"])
            else:
                base64_image = doc["image_base64"]
            content_block = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high",
                },
            }
            score = doc["score"]
            doc_info = {
                "file_path": doc["metadata"]["file_path"],
                "page": doc["metadata"]["page"],
            }
            documents_info.append((doc_info, score))
            content.append(content_block)
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=self.messages
            + [
                {"role": "user", "content": content},
            ],
            temperature=0.0,
        )
        response_content = response.choices[0].message.content
        if return_docs:
            return response_content, documents_info

        return response_content


class CoPaliDualVectorStoreRAGAgent:

    def __init__(
        self,
        model,
        copali_vector_store,
        text_chunk_vector_store,
        server_size_db=False,
        chain_of_thought=False,
    ):
        self.llm_client = AzureOpenAI()
        self.system_prompt = qa_system_prompt
        self.chain_of_thought = chain_of_thought
        self.messages = [{"role": "system", "content": self.system_prompt}]
        if chain_of_thought:
            image = encode_image_to_base64(
                "analysis_results/cropped_regions/figures/001.jpg"
            )
            self.messages += [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the 25th percentile chemistry input to compare fo Sodium",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image}",
                                "detail": "high",
                            },
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": """For this question the information from the table is required. 
                            The 25th percentile is on row 3. Sodium has symbol Na so looking 
                            down the column for Na to row 3 gives the answer 22.4. So the final
                            answer is 22.4""",
                        },
                        {
                            "type": "text",
                            "text": "22.4",
                        },
                    ],
                },
            ]
        self.server_size_db = server_size_db
        self.llm = AzureChatOpenAI(model=model)
        self.embeddings = AzureOpenAIEmbeddings(azure_deployment=EMBEDDING_MODEL)
        self.image_vector_store = ColPaliLanceDBStore(
            copali_vector_store,
            persist_directory=f"./colpali_lancedbs",
            table_name="documents",
            store_images_as_base64=True,
        )
        self.vector_store = self.image_vector_store
        self.text_vector_store = Chroma(
            collection_name=text_chunk_vector_store,
            embedding_function=self.embeddings,
            persist_directory="vector_stores",
        )

    def ask(self, question, return_docs=False):
        content = [{"type": "text", "text": question}]
        documents_info = []

        docs = self.image_vector_store.query(question, n_results=2, server_size_db=True)
        for doc in docs:
            if self.server_size_db:
                base64_image = encode_image_to_base64(doc["image_path"])
            else:
                base64_image = doc["image_base64"]

            content_block = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high",
                },
            }
            score = doc["score"]
            doc_info = {
                "text": "image",
                "score": score,
                "file_path": doc["image_path"],
                "page": doc["image_path"].split(".pdf/page_")[1].split("/figures")[0],
                "image_name": doc["image_name"],
            }
            documents_info.append((doc_info, score))
            content.append(content_block)

        text_docs = self.text_vector_store.similarity_search_with_score(question, k=2)
        for doc, score in text_docs:
            content_block = {"type": "text", "text": doc.page_content}
            doc_info = {"text": doc.page_content, "score": score}
            documents_info.append((doc_info, score))
            content.append(content_block)

        response = self.llm_client.chat.completions.create(
            model="gpt4o",
            messages=self.messages
            + [
                {"role": "user", "content": content},
            ],
            temperature=0.0,
        )
        response_content = response.choices[0].message.content
        if return_docs:
            return response_content, documents_info

        return response_content
