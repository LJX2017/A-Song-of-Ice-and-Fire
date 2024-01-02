import textwrap
from sentence_transformers import SentenceTransformer
import chromadb
chroma_client = chromadb.PersistentClient(path="embeddings/ASoIaF")
collection = chroma_client.get_or_create_collection(name="ASoIaF")
model = SentenceTransformer('thenlper/gte-large-zh')
def load_plot():
    count = 1
    for i in range(1,6):
        with open(f"books/book{i}.txt", "r", encoding="utf-8") as file:
            docs = textwrap.wrap(file.read(), width=300)
        # for c in list(string.ascii_lowercase + string.ascii_uppercase):
        #     data = data.replace(c, "")
        # docs = load_string(data)
        for doc in docs:
            sentence_embeddings = model.encode([doc])
            collection.add(
                embeddings=sentence_embeddings,
                documents=doc,
                ids="plotid" + str(count)
            )
            count += 1
            print(count, doc)
            print("\n\n")
load_plot()