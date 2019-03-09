import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

meta = pd.read_csv('dataset/meta.csv')

index = meta['id']
star = meta['star']

del meta

def clean_review(review):
    review = review.lower()

    review = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', review, flags=re.MULTILINE)
    review = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', review)
    review = re.sub(r'\b[0-9]+\b\s*', '', review)

    review = word_tokenize(review)

    review = [t for t in review if t.isalpha()]

    stop_words = set(stopwords.words('english'))
    review = [t for t in review if not t in stop_words]

    return review

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

