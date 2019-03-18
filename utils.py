import re
import numpy as np
from keras.layers.embeddings import Embedding
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

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

def text_to_vec(review, model, max_length):
    zeros = np.zeros(100).tolist()
    
    review = clean_review(review)
    
    if len(review) >= 80:
        review = review[0:80]
    
    
    vector = []
    for word in review:
        if word in model.vocab:
            vector += model['word'].tolist()
        else:
            vector += zeros

    if len(vector) < max_length * 100:
        size = len(vector)
        extraArr = np.zeros((max_length*100) - size).tolist()
        vector += extraArr
    
    return vector

def generate_batch(X, y, batch_size, vocab_length, max_length, no_classes, model):
    counter = 0
    X = X[0: int(len(X) / batch_size) * batch_size]
    y = y[0: int(len(y) / batch_size) * batch_size]
    
    while True:
        tempIndex = X[counter:counter + batch_size]
        reviews = []    
        for rev in tempIndex:
            with open('dataset/reviewstxt/' + str(rev) + '.txt', 'r') as file:
                review = file.read()
                review = text_to_vec(review, model, max_length)
                review = np.array(review).reshape(1, 100*max_length, 1)
                reviews.append(review)
        if len(reviews) <= 0:
          counter = 0
          continue
      
        reviews = pad_sequences(reviews, maxlen=max_length, padding='post')            
        tempStars = y[counter:counter + batch_size]    
        counter = (counter + batch_size) % len(X) 
        
        #assert reviews.shape == (batch_size, max_length), "{} is not matching with {}".format(reviews.shape, (batch_size, max_length))
        #assert tempStars.shape == (batch_size, no_classes), "{} is not matching with {}".format(tempStars.shape, (batch_size, no_classes))    
        
        yield reviews, tempStars