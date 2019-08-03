import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from utils import clean_review

# Max number of words for a single review
MAX_WORDS = 80

# Reading the Tokenizer pickle file
with open(r"dataset/word_tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)
   
# Reading the datset
data = pd.read_csv('dataset/dataset.csv')

# For testing purposes only
#data = data.sample(frac=0.01)

# Selecting the reviews and ratings
X = data['reviews'].values
y = data['ratings'].values

cleaned_X = []

# Iterating through the reviews
for i in range(len(X)):
    if i % 1000 == 0:
        print("--Cleaning {}th review--".format(i))
    cleaned_X.append(clean_review(X[i]))

# Converting the reviews into sequence
X_vec = tokenizer.texts_to_sequences(X)

# Padding the reviews
X_vec_pad = pad_sequences(X_vec, MAX_WORDS, padding='post')

# Stacking the data
dataset = np.hstack((X_vec_pad, y.reshape(-1,1)))

# Saving the dataset as numpy file
np.save('dataset/preprocessed_dataset', dataset)