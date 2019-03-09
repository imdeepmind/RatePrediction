import numpy as np
import pandas as pd

meta = pd.read_csv('dataset/meta.csv')

index = meta['id']
star = meta['star']

del meta

def generate_batch(index, stars, batch_size):
    counter = 0
    while True:
        tempIndex = index[counter:counter + batch_size]
        reviews = []
        for re in tempIndex:
            with open('dataset/reviews/' + str(re) + '.txt', 'r') as file:
                reviews.append(file.read())
        tempStars = stars[counter:counter + batch_size]
        counter = (counter + batch_size) % len(index)
        yield reviews, tempStars
