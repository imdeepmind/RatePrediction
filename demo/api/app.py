from flask import Flask, jsonify, request
from flask_cors import CORS
import keras
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from keras.preprocessing.sequence import pad_sequences
import pickle
import re
from keras.backend import clear_session

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Max number words in reviews
MAX_WORDS = 80


# Loading the word tokenizer
with open(r"data/word_tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)

# Cleaning the reviews
def clean_review(review):
    # Changing to lowercase
    review = review.lower()
    
    # Removing links and other stuffs from string
    review = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', review, flags=re.MULTILINE)
    review = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', review)
    review = re.sub(r'\b[0-9]+\b\s*', '', review)
    
    return review

# Removing stopwords
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

@app.route('/')
def index():
    return jsonify({
        "success" : True
    })

@app.route('/predict/', methods=['POST'])
def predict():
    try:
        # Asking the user to type a review
        data = request.get_json()
        #data = json.loads(data)
        review = data['review']
            
        if review:
            # Cleaning the review
            review = clean_review(review)
            review = remove_stopwords(review)

            # Using the tokenizer to convert the text to sequence
            review_vec = tokenizer.texts_to_sequences([review])

            # Padding sequences    
            review_vec_pad = pad_sequences(review_vec, MAX_WORDS, padding='post')

            # Main Model
            model = keras.models.load_model('data/model.h5')

            # Compiling the model
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Predicting the class
            predict_class = model.predict_classes(review_vec_pad)
                
            # Predicting probs
            predict_prob = model.predict(review_vec_pad)

            # Weird bug i guess
            clear_session()

            return jsonify({
                "success" : True,
                "data": {
                    "class" : str(predict_class[0]+1),
                    "prob" : str(predict_prob[0])
                }
            })
        else:
            return jsonify({
                "success" : False,
                "message": "Please provide a review"
            })
    except Exception as ex:
        print(str(ex))
        return jsonify({
            "success" : False,
            "message": "Something went wrong"
        })
