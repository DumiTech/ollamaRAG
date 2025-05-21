import os
import glob
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


DATA_PATH = os.path.abspath("../data/")  # Directory to upload PDFs
CHROMA_PATH = os.path.abspath("../chroma_db/")  # Directory to store ChromaDB data


# CONFIG
DELETE_PDF = False
EMBEDDING_MODEL = 'nomic-embed-text'
LLM_MODEL = 'mistral-small3.1:24b-instruct-2503-q8_0'
TEMPERATURE = 0


def get_all_pdf_files(data_path=DATA_PATH):
    """Returns a list of all PDF file paths in the data directory."""
    return glob.glob(os.path.join(data_path, "*.pdf"))


def get_file_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_and_split_all_pdfs(pdf_files):
    """Loads and splits all PDFs, adding file name as metadata."""
    all_chunks = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        try:
            documents = loader.load()
        except Exception as e:
            print(f"Error loading {pdf_path}: {str(e)}")
            continue
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=768,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata['file_hash'] = get_file_hash(pdf_path)

            if not chunk.metadata:
                chunk.metadata = {}
            chunk.metadata['file_name'] = os.path.basename(pdf_path)
        all_chunks.extend(chunks)
    print(f"Loaded and split {len(pdf_files)} PDFs into {len(all_chunks)} chunks.")

    if DELETE_PDF:
        for pdf_file in pdf_files:
            os.remove(pdf_file)
            print(f"Ingested and deleted: {pdf_file}")
    return all_chunks


def get_embedding_function(model_name=EMBEDDING_MODEL):
    """Initializes the Ollama embedding function."""
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Initialized Ollama embeddings with model: {model_name}")
    return embeddings


def get_vector_store(embedding_function, persist_directory=CHROMA_PATH):
    """Initializes or loads the Chroma vector store."""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name="main")
    print(f"Vector store initialized/loaded.")
    return vectorstore


def query_rag(chain, question):
    """Queries the RAG chain and prints the response."""
    print("\nQuerying RAG chain...")
    response = chain.invoke(question)
    print("\nResponse:")
    print(response)


def index_chunks_to_chroma(chunks, embedding_model=EMBEDDING_MODEL):
    """Adds chunks to existing Chroma collection"""
    embedding_function = get_embedding_function(embedding_model)
    vectorstore = get_vector_store(embedding_function)
    vectorstore.add_documents(chunks)
    print(f"Added {len(chunks)} chunks to ChromaDB")
    return vectorstore


def initialize_vectorstore():
    
    pdf_files = get_all_pdf_files()
    if pdf_files:
        chunks = load_and_split_all_pdfs(pdf_files)
        return index_chunks_to_chroma(chunks)
    else:
        print("No PDF files found in the data directory.\n")
        embedding_function = get_embedding_function()
        return get_vector_store(embedding_function)


if __name__ == "__main__":
    vectorstore = initialize_vectorstore()

    if len(vectorstore.get()['ids']) == 0:
        raise ValueError("ChromaDB is empty. Add documents first.")

    # Now you can always query the vectorstore, regardless of PDFs
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5}
    )

    # Create RAG chain for general queries
    template = """Answer the question based ONLY on the following context:
    {context}

    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        num_ctx=8192
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"\nInitialized Ollama Chat with model: {LLM_MODEL}")

    while True:
        try:
            query_question = input('\nAsk anything (or type "q" / "quit" to quit): ')
            if query_question.strip().lower() in {"q", "quit", "exit"}:
                print("\nChat closed!")
                break
            if not query_question.strip():
                continue
            query_rag(rag_chain, query_question)
        except KeyboardInterrupt:
            print("\nInterrupted. Exiting gracefully.")
            break
