# Dependencies
import keras 
import gensim
from utils import text_to_vec

def predict(model='rnn'):
    # Word2Vec Model
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True)
    
    # Main Model
    model = keras.models.load_model('model.h5')
    
    # Compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Asking the user to type a review
    review = input('Please type a review: ')
    
    # Converting the review to vector
    processedReview = text_to_vec(review, word2vec_model, 80)
    
    # Predicting the class
    predict_class = model.predict_classes(processedReview)
    
    # Predicting probs
    predict_prob = model.predict(processedReview)
    
    # Printing the class and prob values
    print(predict_class[0] + 1, predict_prob)

if __name__ == '__main__':
    # Running the loop forever
    while True:
        predict('rnn')