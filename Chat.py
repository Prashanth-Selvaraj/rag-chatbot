import numpy as np
import ollama
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import argparse

                                ## Creating a Knowledge base ##
SYSTEM_PROMPT = """
    You are a precise AI assistant. Answer the question using ONLY the provided context but dont use phrases like 
    "Based on the Provided context:". Just start with a normal factual response based on the context.
    If the context does not contain a clear definition, provide a general definition and then relate it to the context. Do not hallucinate."""

PDF_PATHS = ["C:/Users/Admin/Desktop/AIML/LLM/RAG_Research_Paper.pdf","C:/Users/Admin/Desktop/AIML/LLM/RAG_NLP_Tasks.pdf"]
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

    chunks = splitter.split_text(full_text)   #returns a list of chunks of text from the full_text variable
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
    faiss.normalize_L2(chunk_embeddings)               # Removes influence of magnitude on embeddings
    index = faiss.IndexFlatIP(dimension)               # Creates a Vector Database
    index.add(chunk_embeddings)                        # Stores chunk embeddings in my System RAM
    return index, all_chunks, metadata


def retrieve(index, all_chunks, metadata, user_question, k = TOP_K):
    query_embeddings = embed_batchwise([user_question])
    faiss.normalize_L2(query_embeddings)

    distances, indices = index.search(query_embeddings, k)
    retrieved_chunks = [all_chunks[i] for i in indices[0]]
    retrieved_meta = [metadata[i] for i in indices[0]]

    print("\n--Retrieved Chunks--")
    for j, rt in enumerate(retrieved_chunks):
        print(f"[{j + 1}] Score: {distances[0][j] : .4f}  |  {rt[:100]}.....")

    context = "\n\n---\n".join(
        [f"From: {retrieved_meta[i]['Source']} (chunk {retrieved_meta[i]['Chunk_ID']}):\n{retrieved_chunks[i]}"
            for i in range(k)]
    )

    return retrieved_chunks, retrieved_meta, context


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
    This Block of code is used so that I can Add more Pdf files for data from the CLI instead
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
        print("[INFO] No PDFs provided. Please pass at least one PDF using --pdfs")
        exit(1)

    # index, all_chunks, metadata = build_index(PDF_PATHS)
    convo_history = []  # an empty list for storing convserations

    convo_history.append({"role": "system", "content": SYSTEM_PROMPT})

    while True:
        user_question = input("Ask Anything about LLMs: ")

        if user_question == "exit" or user_question == "EXIT":
            print("Understandable, Have a nice day!!😊")
            break

        else:
            retrieved_chunks, retrieved_meta, context = retrieve(index, all_chunks, metadata, user_question)
            prompt = build_prompt(user_question, context)

            convo_history.append({"role": "user", "content": prompt})

            # Injecting the Augmented prompt into the model during Runtime

            response = ollama.chat(model=CHAT_MODEL, messages=convo_history)
            print(response['message']['content'])

            convo_history.append({"role": "assistant", "content": response['message']['content']})


if __name__ == "__main__":
    main()






