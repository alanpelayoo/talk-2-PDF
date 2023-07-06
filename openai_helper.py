import openai
import pandas as pd
import numpy as np
from transformers import GPT2TokenizerFast
from typing import List, Dict, Tuple
import os


openai.api_key = os.environ["openai_key"]


MODEL_NAME = "curie"
DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "

COMPLETIONS_MODEL = "text-davinci-003"
COMPLETIONS_API_PARAMS = {
    "model": COMPLETIONS_MODEL,
}

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
separator_len = len(tokenizer.tokenize(SEPARATOR))

def get_embedding(text: str, model: str) -> List[float]:
    result = openai.Embedding.create(
      model=model,
      input=text)
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> List[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> List[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_doc_embedding(r.content.replace("\n", " ")) for idx, r in df.iterrows()
    }

def load_embeddings(fname: str, actual_file) -> Dict[Tuple[str, str], List[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    df1 = pd.read_csv(actual_file)
    new_df = df1.merge(df, left_index=True, right_index=True)
    max_dim = max([int(c) for c in new_df.columns if c != "title" and c != "heading" and c != 'content' and c != 'tokens'])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in new_df.iterrows()
    }, new_df

def generate_newdfs(df: pd.DataFrame, df1: pd.DataFrame) -> Dict[Tuple[str, str], List[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    new_df = df1.merge(df, left_index=True, right_index=True)
    max_dim = max([int(c) for c in new_df.columns if c != "title" and c != "heading" and c != 'content' and c != 'tokens'])
    return {
           (r.title, r.heading): [r[i] for i in range(max_dim + 1)] for _, r in new_df.iterrows()
    }, new_df

def vector_similarity(x: List[float], y: List[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    """
    return np.dot(np.array(x), np.array(y))

from typing import Dict, List, Tuple

def order_document_sections_by_query_similarity(query: str, contexts: Dict[Tuple[str, str], np.array]) -> List[Tuple[float, Tuple[str, str]]]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities


def construct_prompt(question: str, document_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, document_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, (_, section_index) in most_relevant_document_sections:
        # Add contexts until we run out of space.

       ## This changed
        document_section = df[df.heading == section_index]

        for i, row in document_section.iterrows():
            chosen_sections_len += row.tokens + separator_len
            if chosen_sections_len > MAX_SECTION_LEN:
                break

            chosen_sections.append(SEPARATOR + row.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: Dict[Tuple[str, str], np.array],
    show_prompt: bool = False, temperature=0, max_length=500
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )

    if show_prompt:
        print(prompt)

    COMPLETIONS_API_PARAMS['temperature'] = temperature
    COMPLETIONS_API_PARAMS['max_tokens'] = max_length

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n")
