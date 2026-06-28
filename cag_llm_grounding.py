import os
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Ensure you have OPENAI_API_KEY set in your .env file
# Example .env file:
# OPENAI_API_KEY=your_openai_api_key_here

# URL to fetch context from
CONTEXT_URL = "https://en.wikipedia.org/wiki/Large_language_model"

# --- Helper Functions ---
def load_and_split_documents(url):
    """Loads content from a URL and splits it into chunks."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    return splits

def create_vector_store(splits):
    """Creates a FAISS vector store from document splits."""
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    return vectorstore

def get_retriever(vectorstore):
    """Returns a retriever from the vector store."""
    retriever = vectorstore.as_retriever()
    return retriever

# --- Main CAG Logic ---
def main():
    # 1. Augment Context: Load and prepare external knowledge
    print(f"Loading and splitting documents from {CONTEXT_URL}...")
    document_splits = load_and_split_documents(CONTEXT_URL)
    print(f"Loaded {len(document_splits)} document chunks.")

    # 2. Grounding: Create a retriever for efficient searching
    print("Creating vector store and retriever...")
    vector_store = create_vector_store(document_splits)
    retriever = get_retriever(vector_store)
    print("Retriever created.")

    # 3. LLM and Prompt Setup
    llm = OpenAI(temperature=0.7)

    # Prompt template that includes context from the retriever
    template = """
    You are a helpful AI assistant.
    Use the following pieces of context to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LCEL Chain: Combines retriever, prompt, and LLM
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # --- User Interaction ---
    while True:
        question = input("Ask a question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        print("Generating answer...")
        # Invoke the RAG chain to get a grounded answer
        answer = rag_chain.invoke(question)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
    else:
        main()
