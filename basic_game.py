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
model = genai.GenerativeModel('gemini-pro')
embedding_model = SentenceTransformer('thenlper/gte-large-zh')
chat = genai.GenerativeModel('gemini-pro').start_chat(history=[])
safety_settings=[
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
    ]

def get_query(query: str, n_results=10):
    embedding = embedding_model.encode([query])
    # print("query embedding is", embedding, query)
    results = collection.query(
        query_embeddings=embedding,
        n_results=n_results
    )
    ans = ""
    for text in results["documents"]:
        for sub_text in text:
            ans += sub_text + "\n"
    return ans


def send(message, show=True):
    response = chat.send_message(message, stream=True, safety_settings=safety_settings)
    text = ""
    for chunk in response:
        if show:
            print(chunk.text)
        text += chunk.text
    return text


def sub_question_generator(question) -> list:
    DEFAULT_SUBQUESTION_GENERATOR_PROMPT = f"""
        您是一位专注于将关于《冰与火之歌》的复杂问题分解为更简单、易处理的子问题的人工智能助手。
        用户的问题是"{question}"
        请你将其拆分为数个子问题，每个问题一行
    """
    response = model.generate_content(DEFAULT_SUBQUESTION_GENERATOR_PROMPT, safety_settings=safety_settings)
    sub_questions = response.text.strip().split('\n')
    print("sub_question_generator:", question, sub_questions)
    return sub_questions


# def response_aggregator(question, responses):
#     prompt = """You are an assistant for question-answering tasks.
#                 Use the following pieces of retrieved context to answer the question.
#                 If you don't know the answer, just say that you don't know.
#                 Use three sentences maximum and keep the answer concise."""
def vector_retrival(question):
    context = get_query(question)
    user_prompt = f"""您是一个回答问题的助手。
                    使用您对冰与火之歌的了解和以下检索到的上下文片段来回答问题。
                    如果您不知道答案，只需说不知道即可。
                    最多使用三个句子，并保持回答简洁。
                    问题: {question}
                    上下文: {context}
                    答案:"""
    response = model.generate_content(user_prompt).text
    print("vector_retrival: ", question, response)
    print("retrival context: ", context)
    return response


def response_aggregator(user_question, responses) -> None:
    """Aggregates the responses from the subquestions to generate the final response.
    """
    print("-------> ⭐ Aggregating responses...")
    context = """   使用您对冰与火之歌的了解和以下的相关信息来回答问题。
                    可以详细的整合下方的资料，并给出全面的回答"""
    for i, response in enumerate(responses):
        context += f"\n 参考问题：{response[0]} 参考答案: {response[1]}"
    context += f"\n使用您对冰与火之歌的了解和以下的相关信息来回复{user_question}"
    response = send(context)


def answer_user_input(question) -> None:
    sub_questions = sub_question_generator(question)
    responses = []
    for sub_question in sub_questions:
        Q = sub_question
        A = vector_retrival(sub_question)
        responses.append((Q, A))
    response_aggregator(question, responses)


# sub_question_generator("兰尼斯特家族的势力在乔弗里登基后如何变化？")
# character = input("输入角色名字")
character = "艾莉亚·史塔克"
settings = send(f"您是《冰与火之歌》模拟人生游戏的系统,玩家扮演的角色为{character}, 请生成{character}的人物面板，包括性别、年龄、主要特征（如勇敢、智慧、狡诈）、当前所处的地点（如临冬城、君临）")
#时间顺序上如何解决？ 有哪些通用的框架？ NPC的问题summerize一下，评估指标有哪些？那些东西要注意？
#有哪些成果？经典应用？有哪些有意思的任务？
#相关工作：大模型在游戏里，AI NPC
# intuitive的方式来呈现！ 要点，表格，图片？
# 比较一下效果
#我的调研+发现所存在的问题
#framwork的梳理， 大的 workflow 是什么？ 如何评价 AINPC 的效果？问题有哪些？ 我发现了哪些问题？那些没有被解决？
#summerize一下，以后工作， 问题中挑一到两个突破，
while True:
    send(f"现在你需要根据之前的事件和结果以及书本中的历史走向为{character}生成一个事件，请扮演系统，描述这个事件的相关人物,地点等，把此次事件用一整段文字尽可能绘声绘色的描绘,并等待玩家操控{character}作出的反应")
    # event = send(
    #     "下面是事件人物，地点的相关信息，你可以参考或者忽略它们。 请注意这些信息可能与此次事件无关，请认真辨别 /***" + get_query(
    #         pre_event) + "***/ 注意在/******/中的信息不重要，把此次事件用一整段文字尽可能绘声绘色的描绘，包括人物心理描写，环境描写，侧面描写等")
    decision = input("下面请输入你的选择\n")
    if decision == "exit":
        break
    print("你的选择带来的后果是：")
    answer_user_input(f"对于这个事件，我做出的选择是/*{decision}*/,请生成此次事件的结果和对角色和冰与火之歌的世界产生的影响,具体的说出对某个角色或家族所产生的影响")
