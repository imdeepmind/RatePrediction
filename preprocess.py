import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 80

with open(r"data/word_tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)
   
data = pd.read_csv('data/dataset.csv')
data = data.sample(frac=1)
#data = data.head(5000)

X = data['reviews'].values
y = data['ratings'].values

X_vec = tokenizer.texts_to_sequences(X)

X_vec_pad = pad_sequences(X_vec, MAX_WORDS, padding='post')

dataset = np.hstack((X_vec_pad, y.reshape(-1,1)))

np.save('data/preprocessed_dataset', dataset)