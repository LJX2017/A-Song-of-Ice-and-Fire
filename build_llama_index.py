from llama_index.llms.gemini import Gemini
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import os
from dotenv import load_dotenv
load_dotenv()
embed_model = HuggingFaceEmbedding(model_name='thenlper/gte-large-zh')

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
# load data
pg_essay = SimpleDirectoryReader(input_dir="./books/", recursive=True).load_data()
print(pg_essay)
service_context = ServiceContext.from_defaults(
    llm=Gemini(),
    callback_manager=callback_manager,
    embed_model=embed_model,
    chunk_size=300
)
# build index and query engine
VectorStoreIndex = VectorStoreIndex.from_documents(
    pg_essay, use_async=True,service_context=service_context
)
VectorStoreIndex.storage_context.persist(persist_dir="llama_index_file")