import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import joblib 
import requests

def create_embeddings(text_list) :
    r = requests.post("http://localhost:11434/api/embed", json={
        "model" : "bge-m3",
        "input" : text_list
    })
    embeddings = r.json()['embeddings']
    return embeddings


def inference(prompt) :
    r = requests.post("http://localhost:11434/api/generate", json={
        "model" : "llama3.2",
        "prompt" : prompt,
        "stream" : False
    })

    response = r.json() 
    print(response)
    return response 


df = joblib.load("embeddings.joblib")

# Taking the user query and generating the query embedding.
incoming_query = input("Enter the Query : ")
query_embedding = create_embeddings([incoming_query])[0]

# finding similarities of query_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
similarities = cosine_similarity(np.vstack(df['embedding']) ,[query_embedding]).flatten()
# print(similarities)
top_results = 5
max_index = similarities.argsort()[::-1][0:top_results]
# print(max_index)
new_df = df.loc[max_index]
# print(new_df)
# print(new_df[['title','number','text']])

prompt = f'''Here are video chunks containing video title, video number, start time, end time in seconds , the text at that time : 

{new_df[['title','number','text','start','end']].to_json(orient='records')}

------------------------------------------------------------------------------------------------------------

" {incoming_query} "

User asked this question related to the video chunks, you have to answer in a human way (dont mention the abouve format , its just for you ) where and how much content is taught in which video (in which video and at what time stamp) and guide the user to go to that particcular video. If user asks unrelated question , tell user user to ask appropriate questions only.
'''

with open("prompt.txt","w") as f :
    f.write(prompt)

response = inference(prompt)['response']
print(response)

with open("response.txt" , "w") as f :
    f.write(response)

# for index, item in new_df.iterrows() :
#     print(index , item["title"], item["number"] , item['text'])