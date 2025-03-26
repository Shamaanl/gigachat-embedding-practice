from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_gigachat.embeddings import GigaChatEmbeddings
import json
import numpy
from json import JSONEncoder
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from config import API_TOKEN

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

with open('result.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()

documents = []

for i in range(len(data)):
    line = data[i].split('/')
    documents.append(Document(
        page_content=line[0],
        metadata={"author": line[2], "ratings": line[1]},
    ))

vectorstore = Chroma.from_documents(
    documents,
    embedding = GigaChatEmbeddings(
    credentials=API_TOKEN, scope="GIGACHAT_API_PERS", verify_ssl_certs=False
    ),
)

vectorstore_data = vectorstore.get(include=["embeddings", "metadatas"])
embs = vectorstore_data["embeddings"]
numpyData = {"array": embs}
encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
file = open('data.json','w',encoding='utf-8')
file.write(encodedNumpyData)
vectorstore_main = vectorstore.get()
with open('embeddings.json', 'w', encoding='utf-8') as ebobo:
    json.dump(vectorstore_main, ebobo, ensure_ascii=False)