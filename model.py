import numpy as np
import pandas as pd
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 50000
MAX_LENGTH = 50
NO_OF_CLASSES = 5
meta = pd.read_csv('dataset/meta.csv')

index = meta['id']
stars = meta['star']

del meta

def generate_batch(index, stars, batch_size):
    counter = 0
    
    while True:
        tempIndex = index[counter:counter + batch_size]
        reviews = []
        
        for rev in tempIndex:
            with open('dataset/reviews/' + str(rev) + '.txt', 'r') as file:
                review = file.read()

                review_hot = one_hot(review, VOCAB_SIZE)
                
                reviews.append(review_hot)

        reviews = pad_sequences(reviews, maxlen=MAX_LENGTH, padding='post')
                
        tempStars = stars[counter:counter + batch_size]
        
        counter = (counter + batch_size) % len(index)
        
        yield reviews, tempStars

