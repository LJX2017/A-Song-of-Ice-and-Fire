from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import os
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import json
with open("a song of ice and fire.json", "r", encoding="utf-8") as file:
    data_dict = json.load(file)

docs = ''.join(list(data_dict.values()))
with open("chinese_db.txt", "w", encoding="utf-8") as file:
    file.write(docs)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.create_documents([docs])
# len(texts)
# print(texts[0].page_content)
# query_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# db = Chroma.from_documents(texts, query_embeddings)
# query = "Lannister"
# query_embeddings.embed_query(query)
# print(db.similarity_search(query))
# def make_qa_chain():
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
#     return RetrievalQA.from_chain_type(
#         llm,
#         retriever=db.as_retriever(),
#         return_source_documents=True
#     )
# qa_chain = make_qa_chain()
# q = "according to the information we have, come up with the future plot of the book series"
# result = qa_chain({"query": q})
# print(result["result"],"\nsources:",)
# for idx, elt in enumerate(result["source_documents"]):
#     print(elt.page_content)