import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
import utils as myutils
import pickle 

meta = pd.read_csv('dataset/meta.csv')
meta = meta.sample(frac=1)

# To limit the dataset. Useful for testing
# meta = meta.head(200000)

ids = meta['id'].values
ratings = pd.get_dummies(meta['star']).values

del meta

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

X_train, X_test, y_train, y_test = train_test_split(ids, ratings, test_size=0.1, random_state=1969)

del ids, ratings

wordToIndex, indexToWord, wordToGlove = myutils.readGloveFile("glove.6B.50d.txt")
pretrainedEmbeddingLayer = myutils.createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, False)

VOCAB_SIZE = len(wordToIndex) + 1  
MAX_LENGTH = 80
NO_OF_CLASSES = 5
BATCH_SIZE = 64
STEPS_PER_EPOCH = int(len(X_train)/BATCH_SIZE)
VALIDATION_STEPS = int(len(X_test)/BATCH_SIZE)

train_gen = myutils.generate_batch(X=X_train, y=y_train, batch_size=BATCH_SIZE, vocab_length=VOCAB_SIZE, max_length=MAX_LENGTH, no_classes=NO_OF_CLASSES, t=tokenizer)
test_gen = myutils.generate_batch(X=X_test, y=y_test, batch_size=BATCH_SIZE, vocab_length=VOCAB_SIZE, max_length=MAX_LENGTH, no_classes=NO_OF_CLASSES, t=tokenizer)

model = Sequential()
model.add(pretrainedEmbeddingLayer)
model.add(LSTM(32))
model.add(Dense(NO_OF_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit_generator(train_gen,
                    epochs=2,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=test_gen,
                    validation_steps=VALIDATION_STEPS)

model.save('model_rnn.h5')