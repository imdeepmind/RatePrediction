from keras.preprocessing.text import Tokenizer
import pandas as pd
from utils import clean_review
import pickle

meta = pd.read_csv('dataset/meta.csv')

ids = meta['id'].values

del meta

tokenizer = Tokenizer()

for id in ids:
    with open('dataset/reviewstxt/' + str(id) + '.txt', 'r') as file:
        review = clean_review(file.read())            
        tokenizer.fit_on_texts(review)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)