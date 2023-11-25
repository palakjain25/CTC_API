#libraries
from fastapi import FastAPI
from pydantic import BaseModel
import ast
import pandas as pd
import numpy as np
import random
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer

app=FastAPI()

class ScoringItem(BaseModel):
    input_chord: str
    input_chord_sequence: int
    input_creativity: int

model = load_model('CTC.h5')
dataset= pd.read_csv('CTC_API_Dataset.csv')

text=list(dataset.new_chord_list.values)
joined_text= " ".join(text)
partial_chords=joined_text[:10000000]
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_chords)
unique_tokens=np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
n_words=1
input_words = []
next_words = []

for i in range(len(tokens)-n_words):
  input_words.append(tokens[i:i + n_words])
  next_words.append(tokens[i+n_words])

def predict_next_chord(input_text, n_best):
  X = np.zeros((1,n_words, len(unique_tokens)))
  for i, word in enumerate(input_text.split()):
    X[0,i,unique_token_index[word]]=1

  predictions = model.predict(X)[0]
  filtered_predictions = np.argpartition(predictions, -n_best)[-n_best:]
  return filtered_predictions.tolist()

chord_match_list = [
    "C", "D", "E", "F", "G", "A", "B",
    "Cm", "Dm", "Em", "Fm", "Gm", "Am", "Bm",
    "Db", "Eb", "Gb", "Ab", "Bb",
    "Dbm", "Ebm", "Gbm", "Abm", "Bbm"
]

def generate_chord(input_text, text_length, creativity=3):
    global chord_match_list  # Access the global variables

    word_sequence = input_text.split()
    current = 0
    used_choices = set(word_sequence)  # Initialize with input_text words to ensure they are not repeated

    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence))[current:current + n_words])
        try:
            # Skip used choices and choices in input_text while generating new choices
            possible = predict_next_chord(sub_sequence, creativity)
            filtered_choices = [idx for idx in possible if unique_tokens[idx] not in used_choices]

            # If there are still choices after filtering, select one; otherwise, choose randomly
            if filtered_choices:
                choice = unique_tokens[random.choice(filtered_choices)]
            else:
                choice = random.choice(unique_tokens)

            # Check if the predicted chord is present in chord_match_list
            predicted_chord = choice
            if predicted_chord in [match_chord for match_chord in chord_match_list]:
                used_choices.add(choice)
            else:
                # If not present, choose a random chord from chord_match_list
                choice = random.choice(chord_match_list)
                used_choices.add(choice)
        except:
            choice = random.choice(unique_tokens)

        word_sequence.append(choice)
        current += 1

    return word_sequence


@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    # Access input_genre and input_chord_sequence from the item parameter
    input_chord = item.input_chord
    input_chord_sequence = item.input_chord_sequence
    input_creativity = item.input_creativity
    result = generate_chord(input_chord, input_chord_sequence, input_creativity)
    return result

