import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Dropout, GlobalMaxPool1D, Bidirectional
from keras.layers.embeddings import Embedding

with open(r"data/word_tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)

MAX_WORDS = 80
NO_OF_CLASSES = 5
VOCAB_SIZE = len(tokenizer.word_index) + 1

dataset = np.load('data/preprocessed_dataset.npy')

X = dataset[:, 0:80]
y = dataset[:, 80]
y = pd.get_dummies(y).values

del dataset

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1, 
                                                    random_state=1969)

del X, y

model = Sequential()
model.add(Embedding(VOCAB_SIZE, 32, input_length=MAX_WORDS))
model.add(Bidirectional(CuDNNLSTM(64, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(40, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(NO_OF_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128)

model.save('data/model.h5')
