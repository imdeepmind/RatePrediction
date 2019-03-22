import pandas as pd
import numpy as np
import pickle
from utils import clean_reviews, text_to_vec
from keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 80

with open(r"data/word_tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)
   
data = pd.read_csv('data/dataset.csv')
data = data.sample(frac=1)
data = data.head(5000)

X = data['reviews'].values
y = data['ratings'].values

cleaned_X = []

for i in range(len(X)):
    if i % 1000 == 0:
        print('--Processing {}th review'.format(i))
    
    vec = text_to_vec(X[i], tokenizer)
    
    vec = pad_sequences(vec, MAX_WORDS, padding='post')
    
    cleaned_X.append(vec)


dataset = pd.DataFrame(columns = ['reviews', 'rating'])

dataset['reviews'] = cleaned_X
dataset['ratings'] = y.tolist()

dataset.to_csv('data/preprocessed_dataset.csv', index=False)