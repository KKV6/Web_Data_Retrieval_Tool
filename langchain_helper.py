import langchain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import MapReduceDocumentsChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

llm = ChatGroq(model="llama3-70b-8192", temperature=0.5, groq_api_key=groq_api_key)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


def process_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    docs = text_splitter.split_documents(data)
    return docs

def create_vector_db(docs):
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local("faiss_index")

def get_qa_chain():
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    retriever = vector_db.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    return chain

if __name__ == "__main__":
    urls = [
        "https://indianexpress.com/section/sports/cricket/live-score/india-vs-new-zealand-final-odi-live-score-full-scorecard-highlights-icc-champions-trophy-2025-innz03092025255197/",
        "https://apnews.com/article/india-new-zealand-cricket-champions-trophy-final-d36fb7f4ec4845c02daddce01c9a696a",
        "https://sportstar.thehindu.com/cricket/champions-trophy/india-wins-champions-trophy-2025-ind-vs-nz-final-match-report-score-highlights/article69310397.ece",
    ]
    docs = process_urls(urls)
    create_vector_db(docs)
    chain = get_qa_chain()
    result = chain.invoke({"question": "Who won the ICC Champions Trophy 2025?"}, return_only_outputs=True)
    print(result['answer'])
