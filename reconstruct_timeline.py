import os
import pandas
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro')
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

df = pandas.read_csv("ASOIAF_Timeline.csv")
TOTAL_EVENTS = 447
YEAR = "Year"
DATE = "Month/Day"
EVENT = "Event"
CHAPTER = "Chapter"
BOOK = "Book"

def generate_context(start: int, finish: int):
    # events = df["Events"].iloc[start: finish+1]
    prompt = ""
    for i in range(start, finish):
        prompt += f"At {df[DATE].iloc[i]} {df[YEAR].iloc[i]}, event \"{df[EVENT].iloc[i]}\" from chapter \"{df[CHAPTER].iloc[i]}\" book \"{df[BOOK].iloc[i]}\"\n"
    return prompt


def generate_summery(start: int, finish: int):
    context = generate_context(start, finish)
    prompt = (f"context: ```{context}``` Synthesize these events to retell them with the aid of your own knowledge. "
              f"These events are separate so be careful to tell them apart")
    return send(prompt)


def send(message, show=True):
    response = chat.send_message(message, stream=True, safety_settings=safety_settings)
    text = ""
    for chunk in response:
        if show:
            print(chunk.text, end="")
        text += chunk.text
    return text


if __name__ == "__main__":
    send("You are a narrator for a game that takes the setting of the world A Song of Ice and Fire. You will write  in a third person and objective manner.", show=False)
    # generate_summery(1,20)
    i = 0
    gap = 20
    plot = ""
    file_name = "plot_summery_v4_20_line.txt"
    while True:
        if i + gap < TOTAL_EVENTS:
            plot += generate_summery(i, i+gap)
        else:
            plot += generate_summery(i, TOTAL_EVENTS)
            break
        i += gap
    path = Path(file_name)
    if not path.exists():
        path.touch()
    with open(file_name,"w") as file:
        file.write(plot)
