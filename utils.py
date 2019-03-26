import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet

def get_pos(review):
    pos = []
    for word in review:
        w, p = nltk.pos_tag([word])[0]
        if p.startswith('J'):
            pos.append((w, wordnet.ADJ))
        elif p.startswith('V'):
            pos.append((w, wordnet.VERB))
        elif p.startswith('N'):
            pos.append((w, wordnet.NOUN))
        elif p.startswith('R'):
            pos.append((w, wordnet.ADV))
        else:
            pos.append(('',''))
    
    return pos
    
def clean_review(review, instring=True):
    # Changing to lowercase
    review = review.lower()
    
    # Removing links and other stuffs from string
    review = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', review, flags=re.MULTILINE)
    review = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', review)
    review = re.sub(r'\b[0-9]+\b\s*', '', review)
    
    # Tokenizing the review
    review = word_tokenize(review) 
    
    # Removing numbers
    review = [t for t in review if t.isalpha()] 
    
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    review = [t for t in review if not t in stop_words] 
    
    lemmatizer = WordNetLemmatizer() 
    lemmatized_review = []
    for word in review:
        w,p = get_pos([word])[0]
        if p != '':
            w = lemmatizer.lemmatize(word, pos=p)
        else:
            w = lemmatizer.lemmatize(word)
        lemmatized_review.append(w)
        
    
    # Returning string
    if instring:
        return ' '.join(lemmatized_review)
    
    # Returning reviewas array
    return lemmatized_review

#def clean_reviews(reviews):
#    # Initialing an array to clean
#    counter = 0
#    cleaned_reviews = []
#    
#    # Looping through the reviews
#    for review in reviews:
#        if counter % 1000 == 0:
#            print('Processing {}th review'.format(counter))
#        
#        if review != None:
#            review = clean_review(review)
#            review = remove_stopwords(review)
#        else:
#            review = ''
#        
#        # Appending the cleaned review
#        cleaned_reviews.append(review)
#        
#        counter += 1
#    
#    # Returning from the method
#    return cleaned_reviews

def text_to_vec(review, tokenizer):
    # Cleaning the review
    if review is None:
        review = ''
    review = clean_review(review)
    
    # Returning the a vector of the review
    return tokenizer.texts_to_sequences(review)
