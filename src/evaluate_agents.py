import json
import os
from time import time
from agents import (
    SimpleRAGAgent,
    AnswerCheckerAgent,
    ImageDescriptionRAGAgent,
    CoPaliRAGAgent,
    CoPaliDualVectorStoreRAGAgent,
)
from langchain_core.documents import Document


def load_questions(file_path):
    with open(file_path, "rb") as f:
        data = json.load(f)
    return data["question_set"]


def main(agent, human_eval=True):
    answer_checker_agent = AnswerCheckerAgent(model="gpt-4.1-test")
    question_set = load_questions("question_set.json")
    results = []
    for i, example in enumerate(question_set):
        question = example["question"]
        expected_answer = example["answer"]
        question_type = example["ground_truth_source"]["type"]
        question_page = example["ground_truth_source"]["page"]
        question_doc = example["ground_truth_source"]["document"]
        answer, docs = agent.ask(question, return_docs=True)

        # print("\n------------------")
        # print("Question:", question)
        # print("Type:", question_type)
        # print("Ground Truth:")
        # print("\tDoc:", question_doc)
        # print(f"\tPage: {question_page}")
        # print("\tAnswer:", expected_answer)
        # print("LLM Answer:", answer)
        # print("RAG Docs:")
        rag_docs = []
        document_retrieved = 0
        for doc, score in docs:
            if isinstance(doc, Document):
                doc = doc.metadata
            try:
                # print(f'\t{doc["file_path"]} - page {doc["page"]}: Score {score}')
                if question_doc in doc["file_path"]:
                    if int(doc["page"]) == int(question_page):
                        document_retrieved = 1

                rag_docs.append(
                    {
                        "file_path": doc["file_path"],
                        "page": doc["page"],
                        "score": score,
                    }
                )
            except KeyError:
                if question_type == "text":
                    document_retrieved = "unknown"
                rag_docs.append(doc)

        # print("Document Found:", document_retrieved)

        llm_judgement = answer_checker_agent.check(question, expected_answer, answer)
        print(f"Q{i} LLM Judgement:", llm_judgement)
        if human_eval:
            human_eval_correct = input("Correct [y/n]?")
        else:
            human_eval_correct = None

        results.append(
            {
                "question_type": question_type,
                "question": question,
                "ground_truth": {
                    "document": question_doc,
                    "page": question_page,
                    "answer": expected_answer,
                },
                "llm_answer": answer,
                "rag_docs": rag_docs,
                "llm_correct": llm_judgement.correct_answer,
                "llm_similarity": llm_judgement.similarity_score,
                "document_retrieved": document_retrieved,
                "human_eval_correct": human_eval_correct,
            }
        )
    vector_store = agent.vector_store._collection_name
    results_dir = f"results/{vector_store}"
    os.makedirs(results_dir, exist_ok=True)
    file_name = str(int(time()))
    if agent.chain_of_thought:
        file_name += "-cot"

    output_file = f"results/{vector_store}/{file_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f)

    return output_file


if __name__ == "__main__":
    vector_stores_to_test = [
        # "simple_text_1752332748",
        # "per_page_text_1752332748",
        # "simple_markdown_1752333450",
        # "per_page_markdown_1752333450",
        # "per_page_text_1752418469",
        # "per_page_markdown_1752418469",
    ]
    agents = [
        # Chunked Text Baseline
        SimpleRAGAgent(
            model="gpt4o",
            vector_store="simple_text_1752332748",
        ),
        # Chunked Text Baseline + CoT
        SimpleRAGAgent(
            model="gpt4o",
            vector_store="simple_text_1752332748",
            chain_of_thought=True,
        ),
        # Chunked Markdown
        SimpleRAGAgent(
            model="gpt4o",
            vector_store="simple_markdown_1752333450",
        ),
        # Chunked Markdown + CoT
        SimpleRAGAgent(
            model="gpt4o",
            vector_store="simple_markdown_1752333450",
            chain_of_thought=True,
        ),
        # Per Page Text
        SimpleRAGAgent(
            model="gpt4o",
            vector_store="per_page_text_1754129274",
        ),
        # Per Page Text + CoT
        SimpleRAGAgent(
            model="gpt4o",
            vector_store="per_page_text_1754129274",
            chain_of_thought=True,
        ),
        # Per Page Markdown
        SimpleRAGAgent(
            model="gpt4o",
            vector_store="per_page_markdown_1754129275",
        ),
        # Per Page Markdown + CoT
        SimpleRAGAgent(
            model="gpt4o",
            vector_store="per_page_markdown_1754129275",
            chain_of_thought=True,
        ),
        # Table Captioning
        ImageDescriptionRAGAgent(
            model="gpt4o",
            vector_store="multi_modal_descriptions_1755713039",
        ),
        # Table Captioning + CoT
        ImageDescriptionRAGAgent(
            model="gpt4o",
            vector_store="multi_modal_descriptions_1755713039",
            chain_of_thought=True,
        ),
        # Per Page CoPali
        CoPaliRAGAgent(
            "gpt4o",
            "full_1753982580",
            server_size_db=True,
        ),
        # Per Page CoPali + CoT
        CoPaliRAGAgent(
            "gpt4o",
            "full_1753982580",
            server_size_db=True,
            chain_of_thought=True,
        ),
        # Dual Vector Store CoPali
        CoPaliDualVectorStoreRAGAgent(
            "gpt4o",
            "chunked_copali_1754149588",
            server_size_db=True,
        ),
        # Dual Vector Store CoPali + CoT
        CoPaliDualVectorStoreRAGAgent(
            "gpt4o",
            "chunked_copali_1754149588",
            server_size_db=True,
            chain_of_thought=True,
        ),
    ]

    results_files = []
    for agent in agents:
        agent_name = agent.vector_store._collection.name
        if agent.chain_of_thought:
            agent_name += " + CoT"
        print(agent_name)
        output_file = main(agent, human_eval=False)
        results_files.append(output_file)

    print(results_files)
