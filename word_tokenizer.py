# Dependencies
import pickle
import pandas as pd
from utils import clean_review, remove_stopwords
from keras.preprocessing.text import Tokenizer

# Max number of words
MAX_WORDS = 10000

# Initializing the tokenizer
tokenizer = Tokenizer(num_words=MAX_WORDS)
    
# Reading the dataset
path = 'data/dataset.csv'
data = pd.read_csv(path)

# For now I'll train the tokenizer only on 50000 reviews
data = data.head(50000)

# We only need the reviews column
reviews = data['reviews'].values

# Just printing the length of the array
print('There are total {} reviews in the dataset'.format(len(reviews)))

# Deleting data variable to save memmory
del data

# Looping through the ararys 
for i in range(len(reviews)):
    # Basic cleaning of the text
    review = clean_review(reviews[i])
    review = remove_stopwords(review)
    
    # Fitting the tokenizer
    tokenizer.fit_on_texts([review])
    
    # Better to see something in the console
    if i % 1000 == 0:
        print('--Processing {}th review--'.format(i))
        
# Saving memory
del reviews

# Saving the tokenizer as a pickle file
with open('data/word_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)