import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import openai
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY not found in environment variables.")
else:
    print(f"OPENROUTER_API_KEY loaded: {OPENROUTER_API_KEY[:5]}...")


def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

import time

def get_vector_store(text_chunks):
    # Limit to first 1 chunk to avoid Rate Limit on free tier for testing
    if len(text_chunks) > 1:
        print(f"Warning: Truncating {len(text_chunks)} chunks to 1 to avoid rate limits.")
        text_chunks = text_chunks[:1]
    
    print("Waiting 10 seconds to cool down API...")
    time.sleep(10)
        
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0.3,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def test_agent():
    pdf_path = r"C:\Users\shrik\Desktop\LLM\TestDoc.pdf"
    print(f"Processing {pdf_path}...")
    
    # 1. Extract Text
    raw_text = get_pdf_text(pdf_path)
    print(f"Text extracted. Length: {len(raw_text)} chars")

    # 2. Split Text
    text_chunks = get_text_chunks(raw_text)
    print(f"Text split into {len(text_chunks)} chunks.")

    # 3. Create Vector Store
    print("Creating vector store (this uses Embeddings API)...")
    vector_store = None
    try:
        vector_store = get_vector_store(text_chunks)
        print("Vector store created.")
    except Exception as e:
        print(f"\nError creating vector store (likely Rate Limit or Model not found): {e}")
        print("Falling back to direct context (sending text directly to LLM)...")
    # print("Skipping vector store creation to test Chat API directly.")

    # 4. Define Questions
    questions = [
        "What is the main topic of this document?",
        "Summarize the key points.",
        "Is there any mention of 'specific details'?" 
    ]

    # 5. Ask Questions
    chain = get_conversational_chain()
    
    print("\n--- Starting Q&A ---\n")
    
    for q in questions:
        print(f"Question: {q}")
        if vector_store:
            docs = vector_store.similarity_search(q)
        else:
            # Fallback: Use first 10k chars as context
            from langchain.docstore.document import Document
            # limit to 10k chars to be safe for context window
            context_text = raw_text[:10000]
            docs = [Document(page_content=context_text)]
            
        try:
            response = chain({"input_documents": docs, "question": q}, return_only_outputs=True)
            print(f"Answer: {response['output_text']}\n")
        except Exception as e:
            print(f"Error generating answer: {e}")
        print("-" * 30)

if __name__ == "__main__":
    test_agent()
