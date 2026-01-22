import requests 
import os
import json
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib

def create_embeddings(text_list) :
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3",
        "input" : text_list
    })
    embeddings = r.json()['embeddings']
    return embeddings

jsons = os.listdir("jsons")
chunk_id = 0
my_dicts = []
for json_file in  jsons : 
    with open(f'jsons/{json_file}','r') as f :
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embeddings([c['text'] for c in content['chunks']])
    for i,chunk in enumerate(content['chunks']) :
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
    print(f"Completed Embeddings for {json_file}")

# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
# saVE this dataframe 
joblib.dump(df,'embeddings.joblib')

