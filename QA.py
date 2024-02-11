from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
import chromadb
chroma_client = chromadb.PersistentClient(path="embeddings/ASoIaF")
collection = chroma_client.get_or_create_collection(name="ASoIaF")
#print(collection.peek())
model = SentenceTransformer('thenlper/gte-large-zh')
def get_query(query: str, n_results=10):
    embedding = model.encode([query])
    #print("query embedding is", embedding, query)
    results = collection.query(
        query_embeddings=embedding,
        n_results=n_results
    )
    for text in results["documents"]:
        for sub_text in text:
            print(sub_text)
    return results
get_query("谁是七国国王？",3)
get_query("兰尼斯特家族",3)

# print(get_query("卧槽",5))