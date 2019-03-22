import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_review(review):
    # Changing to lowercase
    review = review.lower()
    
    # Removing links and other stuffs from string
    review = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', review, flags=re.MULTILINE)
    review = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', review)
    review = re.sub(r'\b[0-9]+\b\s*', '', review)
    
    return review

def remove_stopwords(review, instring=True):
    # Tokenizing the review
    review = word_tokenize(review) 
    
    # Removing numbers
    review = [t for t in review if t.isalpha()] 
    
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    review = [t for t in review if not t in stop_words] 
    
    # Returning string
    if instring:
        return ' '.join(review)
    
    # Returning reviewas array
    return review

def clean_reviews(reviews):
    # Initialing an array to clean
    counter = 0
    cleaned_reviews = []
    
    # Looping through the reviews
    for review in reviews:
        if counter % 1000 == 0:
            print('Processing {}th review'.format(counter))
        
        if review != None:
            review = clean_review(review)
            review = remove_stopwords(review)
        else:
            review = ''
        
        # Appending the cleaned review
        cleaned_reviews.append(review)
        
        counter += 1
    
    # Returning from the method
    return cleaned_reviews

def text_to_vec(review, tokenizer):
    # Cleaning the review
    if review is None:
        review = ''
    review = clean_review(review)
    review = remove_stopwords(review)
    
    # Returning the a vector of the review
    return tokenizer.texts_to_sequences(review)
