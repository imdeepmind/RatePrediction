# TODO: Need to improve this code

import gensim 
import pandas as pd
from utils import clean_review

data = pd.read_csv('dataset/meta.csv')

ids = data['id'].values

del data

reviews = []
for id in ids:
    with open('dataset/reviewstxt/' + str(id) + '.txt', 'r') as file:
        review = clean_review(file.read())            
        reviews.append(review)
        
model = gensim.models.Word2Vec(sentences=reviews, size=100, window=5, workers=4, min_count=1)

model.wv.save_word2vec_format('word2vec.bin', binary=True)
