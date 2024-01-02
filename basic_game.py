import string
from time import sleep

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
import chromadb
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
chroma_client = chromadb.PersistentClient(path="embeddings/ASoIaF")
collection = chroma_client.get_or_create_collection(name="ASoIaF")
model = SentenceTransformer('thenlper/gte-large-zh')
chat = genai.GenerativeModel('gemini-pro').start_chat(history=[])

def get_query(query: str, n_results=10):
    embedding = model.encode([query])
    #print("query embedding is", embedding, query)
    results = collection.query(
        query_embeddings=embedding,
        n_results=n_results
    )
    ans = ""
    for text in results["documents"]:
        for sub_text in text:
            ans += sub_text + "\n"
    return ans


def send(message, show = True):
    response = chat.send_message(message, stream=True, safety_settings=[
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
          "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
          "threshold": "BLOCK_NONE",
        },
        {
          "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
          "threshold": "BLOCK_NONE",
        }
    ])
    text = ""
    for chunk in response:
        if show:
            print(chunk.text)
        text += chunk.text
    return text


start = "请为《冰与火之歌》模拟人生游戏创建一个角色。现在先告诉我角色名字，注意不要输出除了名字以外的任何信息"
character = send(start)
settings = send(f"生成{character}的性别、年龄、主要特征（如勇敢、智慧、狡诈）、当前所处的地点（如临冬城、君临）")
while True:
    pre_event = send(f"现在你需要根据之前的事件和结果为{character}生成一个事件，请告诉我这个事件的相关人物,地点等", False)
    event = send("下面是事件人物，地点的相关信息，你可以参考或者忽略它们。 请注意这些信息可能与此次事件无关，请认真辨别 /***" + get_query(pre_event) + "***/ 注意在/******/中的信息不重要，把此次事件用一整段文字尽可能绘声绘色的描绘，包括人物心理描写，环境描写，侧面描写等")
    dicision = send("根据你刚刚生成的事件，预测此次事件的结果和对角色和冰与火之歌的世界产生的影响,具体的说出对某个角色或家族所产生的影响")
    q = input()
