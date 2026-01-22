import whisper
import json
import os
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = whisper.load_model("small",device=device) 

audios = os.listdir("Audios")

for audio in audios :
    number = audio.split("_")[0]
    title = audio.split("_")[1].split(" .")[0]
    result = model.transcribe(audio=f"Audios/{audio}",language="hi",task="translate",word_timestamps=False,fp16=True if device == "cuda" else False )

    chunks = []
    for segment in result['segments'] :
        chunks.append({'number':number,'title':title,'start':segment['start'],'end':segment['end'],'text':segment['text']})
    
    chunks_with_metadata = {'chunks' : chunks , 'text':result['text']}
    with open (f"jsons/{number}_{title}.json", 'w') as f :
        json.dump(chunks_with_metadata , f)

print('<< Task Successfully Completed >>')