# Dependencies
import keras 
import pickle
from utils import clean_review, remove_stopwords
from keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 80

# Main Model
model = keras.models.load_model('data/model.h5')

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

with open(r"data/word_tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)
    

def predict():

    # Asking the user to type a review
    review = input('Please type a review: ')
    
    review = clean_review(review)
    review = remove_stopwords(review)
    
    review_vec = tokenizer.texts_to_sequences([review])
    
    review_vec_pad = pad_sequences(review_vec, MAX_WORDS, padding='post')
    
     # Predicting the class
    predict_class = model.predict_classes(review_vec_pad)
    
    # Predicting probs
    predict_prob = model.predict(review_vec_pad)
    
    # Printing the class and prob values
    print(predict_class[0] + 1, predict_prob)

if __name__ == '__main__':
    # Running the loop forever
    while True:
        predict()