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
    
    # Changing he'll to he will
    review = re.sub(r"i'm", "i am", review)
    
    review = re.sub(r"aren't", "are not", review)
    review = re.sub(r"couldn't", "counld not", review)
    review = re.sub(r"didn't", "did not", review)
    review = re.sub(r"doesn't", "does not", review)
    review = re.sub(r"don't", "do not", review)
    review = re.sub(r"hadn't", "had not", review)
    review = re.sub(r"hasn't", "has not", review)
    review = re.sub(r"haven't", "have not", review)
    review = re.sub(r"isn't", "is not", review)
    review = re.sub(r"it't", "had not", review)
    review = re.sub(r"hadn't", "had not", review)
    review = re.sub(r"won't", "will not", review)
    review = re.sub(r"can't", "cannot", review)
    review = re.sub(r"mightn't", "might not", review)
    review = re.sub(r"mustn't", "must not", review)
    review = re.sub(r"needn't", "need not", review)
    review = re.sub(r"shouldn't", "should not", review)
    review = re.sub(r"wasn't", "was not", review)
    review = re.sub(r"weren't", "were not", review)
    review = re.sub(r"won't", "will not", review)
    review = re.sub(r"wouldn't", "would not", review)
    
    review = re.sub(r"\'s", " is", review)
    
    review = re.sub(r"\'ll", " will", review)
    
    review = re.sub(r"\'ve", " have", review)
    
    review = re.sub(r"\'re", " are", review)
    
    review = re.sub(r"\'d", " would", review)
    
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
    
    # Lemmatizer
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

def text_to_vec(review, tokenizer):
    # Cleaning the review
    if review is None:
        review = ''
    review = clean_review(review)
    
    # Returning the a vector of the review
    return tokenizer.texts_to_sequences(review)
