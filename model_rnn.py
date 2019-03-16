import pandas as pd
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten, CuDNNLSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from keras import optimizers as op

VOCAB_SIZE = 50000
MAX_LENGTH = 80
NO_OF_CLASSES = 5
BATCH_SIZE = 64

meta = pd.read_csv('meta.csv')
meta = meta.sample(frac=1)

# meta = meta.head(200000)

index = meta['id'].values
stars = pd.get_dummies(meta['star']).values

X_train, X_test, y_train, y_test = train_test_split(index, stars, test_size=0.1, random_state=1969)

length_train = len(X_train)
length_test = len(X_test)                    

def clean_review(review):
    review = review.lower()
    review = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', review, flags=re.MULTILINE)
    review = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', review)
    review = re.sub(r'\b[0-9]+\b\s*', '', review)
    return review

def generate_batch(X, y, batch_size):
    counter = 0
    
    X = X[0: int(len(X) / batch_size) * batch_size]
    y = y[0: int(len(y) / batch_size) * batch_size]
    
    while True:
        tempIndex = X[counter:counter + batch_size]
        reviews = []
        
        for rev in tempIndex:
            with open('reviewstxt/' + str(rev) + '.txt', 'r') as file:
                review = file.read()

                review = clean_review(review)
                
                review = word_tokenize(review)
                
                review = [t for t in review if t.isalpha()]
                stop_words = set(stopwords.words('english'))
                review = [t for t in review if not t in stop_words]
                
                review = " ".join(review)
                
                review_hot = one_hot(review, VOCAB_SIZE)
                
                reviews.append(review_hot)
                
        if len(reviews) <= 0:
          counter = 0
          continue

        reviews = pad_sequences(reviews, maxlen=MAX_LENGTH, padding='post')
                
        tempStars = y[counter:counter + batch_size]
        
        counter = (counter + batch_size) % len(index)
        
        assert reviews.shape == (BATCH_SIZE, MAX_LENGTH), "{} is not matching with {}".format(reviews.shape, (BATCH_SIZE, MAX_LENGTH))
        assert tempStars.shape == (BATCH_SIZE, NO_OF_CLASSES), "{} is not matching with {}".format(tempStars.shape, (BATCH_SIZE, NO_OF_CLASSES))
        
        yield reviews, tempStars

model = Sequential()

model.add(Embedding(VOCAB_SIZE, 64, input_length=MAX_LENGTH))

model.add(CuDNNLSTM(32))

model.add(Dense(NO_OF_CLASSES, activation='softmax'))


# model.compile(loss='categorical_crossentropy', optimizer=op.SGD(lr=0.01), metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=op.RMSprop(lr=2e-5), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

train_gen = generate_batch(X=X_train, y=y_train, batch_size=BATCH_SIZE)
test_gen = generate_batch(X=X_test, y=y_test, batch_size=BATCH_SIZE)

model.fit_generator(train_gen,
                    epochs=2,
                    steps_per_epoch=int(length_train/(BATCH_SIZE)),
                    validation_data=test_gen,
                    validation_steps=int(length_test/(BATCH_SIZE)))


model.save('model_rnn.h5')
