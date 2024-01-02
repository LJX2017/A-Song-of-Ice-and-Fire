import string

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
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import textwrap
import os
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
import chromadb
chroma_client = chromadb.PersistentClient(path="embeddings/ASoIaF")
collection = chroma_client.get_or_create_collection(name="ASoIaF")
def add_plot():
    plot = {1: ""}
def load_json():
    count = 1
    import json
    with open("a song of ice and fire.json", "r", encoding="utf-8") as file:
        data_dict = json.load(file)
    model = SentenceTransformer('thenlper/gte-large-zh')
    for key in data_dict.keys():
        docs = textwrap.wrap(data_dict[key], 300)
        for doc in docs:
            sentence_embeddings = model.encode([doc])
            collection.add(
                embeddings=sentence_embeddings,
                documents=doc,
                ids="id" + str(count)
            )
            count += 1
            print(key, count, doc)
            print("\n\n")
def load_txt():
    count = 1
    with open("background.txt", "r", encoding="utf-8") as file:
        docs = textwrap.wrap(file.read(), width=300)
    # for c in list(string.ascii_lowercase + string.ascii_uppercase):
    #     data = data.replace(c, "")
    model = SentenceTransformer('thenlper/gte-large-zh')
    # docs = load_string(data)
    for doc in docs:
        sentence_embeddings = model.encode([doc])
        collection.add(
            embeddings=sentence_embeddings,
            documents=doc,
            ids="id" + str(count)
        )
        count += 1
        print(count, doc)
        print("\n\n")



load_json()
print("end")
# create the open-source embedding function
# query_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#
# # load it into Chroma
# db = Chroma.from_documents(docs, query_embeddings)
# print("check point 1")
# # query it
# query = "tell me"
# docs = db.similarity_search(query)
#
# # print results
# print(docs[0].page_content)

# def make_qa_chain():
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
#     return RetrievalQA.from_chain_type(
#         llm,
#         retriever=db.as_retriever(search_type="mmr", search_kwargs={"fetch_k":10}),
#         return_source_documents=True
#     )
# qa_chain = make_qa_chain()
# q = "summerize the plot with given information"
# result = qa_chain({"query": q})
# print(result["result"],"\nsources:",)
# for idx, elt in enumerate(result["source_documents"]):
#     print(elt.page_content)