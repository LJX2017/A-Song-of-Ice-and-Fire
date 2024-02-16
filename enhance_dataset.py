import os
import pandas
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


df = pandas.read_csv("ASOIAF_Timeline.csv")
TOTAL_EVENTS = df.shape[0]
# print(TOTAL_EVENTS)
YEAR = "Year"
DATE = "Month/Day"
EVENT = "Event"
CHAPTER = "Chapter"
BOOK = "Book"


def generate_context(start: int, finish: int):
    # events = df["Events"].iloc[start: finish+1]
    prompt = ""
    for i in range(start, finish):
        # print(f"At {df[DATE].iloc[i]} Year {df[YEAR].iloc[i]}, event \"{df[EVENT].iloc[i]}\"\n")
        prompt += f"At {df[DATE].iloc[i]} Year {int(df[YEAR].iloc[i])}, event \"{df[EVENT].iloc[i]}\"\n"
    return prompt

'''
You are an export of the novel series A Song of Ice and Fire.
You are asked to elaborate on each of the events that I will mention below.
Only provide a short paragraph for each event independently.
only generate pure text about the events themselves, and only switch lines after completing one event to separate them.
'''
SYSTEM_MESSAGE = "You are an export of the novel series A Song of Ice and Fire. You are asked to elaborate on each of the events that I will mention below. Only provide a short paragraph for each event independently. only generate pure text about the events themselves, and only switch lines after completing one event to separate them."
from openai import OpenAI
import time
client = OpenAI()

path = Path("enhanced_story_information")
# Initial message to start the conversation
# initial_message = {
#     "model": "gpt-3.5-turbo",
#     "response_format": { "type": "json_object" },
#     "messages": [
#         {"role": "system", "content": SYSTEM_MESSAGE},
#         # {"role": "user", "content": "Tell me about Jon Snow."}
#     ]
# }

# Start the conversation and store the initial response
# response = client.chat.completions.create(**initial_message)
#
subsequent_messages = []
i = 0
gap = 10
plot = ""
while True:
    if i + gap < TOTAL_EVENTS:
        plot = generate_context(i, i + gap)
        subsequent_messages.append(plot)
    else:
        plot = generate_context(i, TOTAL_EVENTS)
        subsequent_messages.append(plot)
        break
    i += gap
# print(subsequent_messages)
for message in subsequent_messages:
    print(message)
#     # Update the request for the next message
#     new_message_request = {
#         "model": "gpt-3.5-turbo",
#         "response_format": { "type": "json_object" },
#         "chat_id": response.data['id'],  # Use chat_id from the previous response
#         "messages": [
#             {"role": "user", "content": message}
#         ]
#     }
#
#     # Reuse the same response variable to store the new response
#     response = client.chat.completions.create(**new_message_request)
#     print(response.choices[0].message.content)
#     with path.open("a") as file:
#         file.write(response.choices[0].message.content + "\n")
#     time.sleep(2)
#
# with path.open("a") as file:

