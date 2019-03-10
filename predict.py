import keras
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 50000
MAX_LENGTH = 50

def process_review(review):
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