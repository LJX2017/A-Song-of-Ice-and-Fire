from llama_index.llms.gemini import Gemini
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext
import time
from llama_index import SimpleDirectoryReader
import os
from dotenv import load_dotenv
load_dotenv()
embed_model = HuggingFaceEmbedding(model_name='thenlper/gte-large-zh')
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
# load data
service_context = ServiceContext.from_defaults(
    llm=Gemini(),
    callback_manager=callback_manager,
    embed_model=embed_model
)
from llama_index import StorageContext, load_index_from_storage
tic = time.time_ns()
# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="llama_index_file")

# load index
Index = load_index_from_storage(storage_context, service_context=service_context)
toc = time.time_ns()
print("load_time is",(toc-tic)/1e6,"ms")
vector_query_engine = Index.as_query_engine()
# print("finished loading index")
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="冰与火之歌维基百科",
            description="中文写的维基百科",
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    service_context=service_context,
    use_async=True,
)
response = query_engine.query(
    "兰尼斯特家族现在和以前比较有什么变化"
)