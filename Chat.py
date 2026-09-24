import numpy as np
import ollama
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import argparse
from sentence_transformers import CrossEncoder
import pickle
import os

                                ## Creating a Knowledge base ##
SYSTEM_PROMPT = """
You are an expert AI/ML mentor helping aspiring developers learn machine learning and AI concepts.

YOUR BEHAVIOUR:
- Use the provided research paper context as your primary source for explanations
- Supplement with your own knowledge to give complete, beginner friendly explanations
- When explaining concepts always follow this structure:
    1. Simple definition
    2. Intuitive analogy
    3. Technical explanation
    4. Real world application
- Always recommend 2-3 reference links for further reading

RULES FOR PROVIDING LINKS:
When providing reference links:
- Only suggest links you are highly confident about
- Prefer these trusted domains: arxiv.org, huggingface.co, pytorch.org, paperswithcode.com
- If you are not sure of the exact URL, suggest the domain and tell the user to search for it
- Never guess or construct URLs — say "search for X on arxiv.org" instead
- Always mention the paper title and authors alongside any link

STRICT RULE ON LINKS:
You are PROHIBITED from constructing or generating any URLs directly.
Instead always say:
"Search for [paper title] by [authors] on arxiv.org"
or
"Find this on huggingface.co by searching [topic]"

The only exception is if the retrieved context contains an exact URL — 
in that case you may use it verbatim.

- Suggest what the learner should study next after this concept
- If the context doesn't cover the question, answer from your own knowledge and say so
- Never hallucinate links — only suggest well known legitimate resources like arxiv.org, huggingface.co, pytorch.org, papers with code
"""

#pdf paths for copy/paste "C:/Users/Admin/Desktop/AIML/LLM/RAG_Research_Paper.pdf","C:/Users/Admin/Desktop/AIML/LLM/RAG_NLP_Tasks.pdf"

CHUNKS_PATH = "C:/Users/Admin/Desktop/AIML/LLM/saved/pickle_dump/all_chunks.pkl"
METADATA_PATH = "C:/Users/Admin/Desktop/AIML/LLM/saved/pickle_dump/metadata.pkl"
INDEX_PATH = "C:/Users/Admin/Desktop/AIML/LLM/saved/index.faiss"
EMBED_MODEL   = "mxbai-embed-large"
CHAT_MODEL    = "gemma3:4b"
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 150
TOP_K         = 5

def load_and_chunk_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {pdf_path}")
        exit(1)

    full_text = ""

    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators = ["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(full_text)          #returns a list of chunks of text from the full_text variable
    return chunks


                        ## Creating Vector Embeddings of the Chunks##

def embed_batchwise(text):
    try:
        response = ollama.embed(model=EMBED_MODEL , input=text)
        return np.array(response['embeddings'], dtype=np.float32)
    except Exception as e:
        print(f"[Error] Embedding failed: {e}")
        exit(1)


def build_index(pdf_files):
    all_chunks = []
    metadata = []

    for pdf in pdf_files:
        chunks = load_and_chunk_pdf(pdf)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            metadata.append({
                "Source": pdf,
                "Chunk_ID": i
            })

    chunk_embeddings = embed_batchwise(all_chunks)

    dimension = chunk_embeddings.shape[1]
    faiss.normalize_L2(chunk_embeddings)               # Removes influence of magnitude on embeddings by bringing the magnitude to the value of 1

    index_exists = os.path.exists(INDEX_PATH)

    if index_exists:
        index = faiss.read_index(INDEX_PATH)                  #Loading existing index

        with open(CHUNKS_PATH, "rb") as f:                    #loading all_chunks.pkl
            existing_chunks = pickle.load(f)

        with open(METADATA_PATH, "rb") as f:                  # loading metadata.pkl
            existing_metadata = pickle.load(f)

        index.add(chunk_embeddings)

        all_chunks = existing_chunks + all_chunks
        metadata = existing_metadata + metadata

    else:
        index = faiss.IndexFlatIP(dimension)               # Creates a Vector Database
        index.add(chunk_embeddings)                        # Stores chunk embeddings in my System RAM

    #Saving the index in local disk for persistence
    faiss.write_index(index, INDEX_PATH)

    #Storing the all_chunks and metadata as pickle files
    with open(CHUNKS_PATH, "wb") as f:                     #converts the all_chunks list into all_chunks.pkl which will contain list of chunks in binary representation
        pickle.dump(all_chunks, f)
    with open(METADATA_PATH, "wb") as f:                   #converts the all_chunks list into metadata.pkl which will contain dictionary of metadata in binary representation
        pickle.dump(metadata, f)

    return index, all_chunks, metadata


def retrieve(index, all_chunks, metadata, user_question, reranker, k=TOP_K):
    query_embeddings = embed_batchwise([user_question])
    faiss.normalize_L2(query_embeddings)

    distances, indices = index.search(query_embeddings, k)                                # performs Semantic Similarity search between the query and the VB chunks
    retrieved_chunks = [all_chunks[i] for i in indices[0]]
    retrieved_meta = [metadata[i] for i in indices[0]]

    scores = reranker.predict([(user_question, chunk) for chunk in retrieved_chunks])     # Returns a list of raw Logit scores between the query and each retrieved chunk in retrieved_chunks
    ranked_indices = np.argsort(scores)[::-1]                                             # sorts the list of indices based on the logit scores in Descending order

    ranked_retrieval = [retrieved_chunks[i] for i in ranked_indices]
    ranked_meta = [retrieved_meta[i] for i in ranked_indices]

    print("\n-----Reranked Chunks------")
    for j, rt in enumerate(ranked_retrieval):
        print(f"[{j + 1}] Score: {scores[ranked_indices[j]]:.4f}  |  {rt[:100]}.....")

    context = "\n\n---\n".join(
        [f"From: {ranked_meta[i]['Source']} (chunk {ranked_meta[i]['Chunk_ID']}):\n{ranked_retrieval[i]}"
         for i in range(k)])

    return ranked_retrieval, ranked_meta, context


def build_prompt(user_question, context):
    ## Augmentation of the Prompt ##
    prompt = f"""
    Context:
    {context}

    Question:
    {user_question}
    """
    return prompt


def main():
    """
    This Block of code is used so that I can Add more Pdf files for data from the CLI instead of
    manually updating them in the code
    """
    parser = argparse.ArgumentParser(description='RAG Chatbot')
    parser.add_argument('--pdfs', nargs="+", required=False, default=None, help="Path to  one or more PDF files")
    args = parser.parse_args()

    """
    Main Functioning Code
    """

    if args.pdfs:
        index, all_chunks, metadata = build_index(args.pdfs)
    else:
        # print("[INFO] No PDFs provided. Please pass at least one PDF using --pdfs")
        # exit(1)
        if not os.path.exists(INDEX_PATH):
            print("[INFO] No saved index found. Please pass at least one PDF using --pdfs")
            exit(1)
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as file:
            all_chunks = pickle.load(file)
        with open(METADATA_PATH, "rb") as file:
            metadata = pickle.load(file)

    convo_history = []  # an empty list for storing convserations

    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')             #creates an instance of CrossEncoder model
    convo_history.append({"role": "system", "content": SYSTEM_PROMPT})         #append the system role before the loop so that the LLM gets instructed early

    while True:
        user_question = input("Ask Anything about LLMs: ")

        if user_question == "exit" or user_question == "EXIT":
            print("Understandable, Have a nice day!!😊")
            break

        else:
            retrieved_chunks, retrieved_meta, context = retrieve(index, all_chunks, metadata, user_question, reranker)
            prompt = build_prompt(user_question, context)

            convo_history.append({"role": "user", "content": prompt})

            # Injecting the Augmented prompt into the model during Runtime

            response = ollama.chat(model=CHAT_MODEL, messages=convo_history)           #Here the convo history acts as the augmented prompt being injected into the LLM as input during runtime
            print(response['message']['content'])

            convo_history.append({"role": "assistant", "content": response['message']['content']})


if __name__ == "__main__":
    main()






