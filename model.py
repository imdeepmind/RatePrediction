import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import CuDNNLSTM, Dense, Dropout, GlobalMaxPool1D, Bidirectional, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding

with open(r"data/word_tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)

MAX_WORDS = 80
NO_OF_CLASSES = 5
VOCAB_SIZE = 10000
EPOCHS = 3
BATCH_SIZE = 1024

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

model.add(Dropout(0.5))

model.add(Conv1D(filters=512, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(3))

model.add(Dropout(0.5))

model.add(Bidirectional(CuDNNLSTM(128, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(10, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(NO_OF_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

model.save('data/model.h5')
