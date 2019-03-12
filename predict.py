import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

VOCAB_SIZE = 50000
MAX_LENGTH = 80

def clean_review(review):
    review = review.lower()
    review = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', review, flags=re.MULTILINE)
    review = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', review)
    review = re.sub(r'\b[0-9]+\b\s*', '', review)
    return review


def process_review(review):
    review = clean_review(review)
                
    review = word_tokenize(review)
    
    review = [t for t in review if t.isalpha()]
    stop_words = set(stopwords.words('english'))
    review = [t for t in review if not t in stop_words]
    
    review = " ".join(review)
    
    review = one_hot(review, VOCAB_SIZE)
    
    review = pad_sequences([review], maxlen=MAX_LENGTH, padding='post')
    
    return review


def predict():
    model = keras.models.load_model('model.h5')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    review = input('Please type a review: ')
    
    processedReview = process_review(review)
    
    predict_class = model.predict_classes(processedReview)
    
    predict_prob = model.predict(processedReview)
    
    print(predict_class[0] + 1, predict_prob)
    

while True:
    predict()
