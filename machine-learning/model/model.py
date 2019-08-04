import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Note: Use LSTM if you dont have a NVDIA  else use CuDNNLSTM
from keras.layers import LSTM, Dense, Dropout, GlobalMaxPool1D, Bidirectional, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Reading the tokenizer pickle file
with open(r"dataset/word_tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)

# Constants for model configuration
MAX_WORDS = 80
NO_OF_CLASSES = 5
VOCAB_SIZE = 10000
EPOCHS = 50
BATCH_SIZE = 1024

# Loading the dataset
dataset = np.load('dataset/preprocessed_dataset.npy')

# Spliting into X and y
X = dataset[:, 0:80]
y = dataset[:, 80]
y = pd.get_dummies(y).values

# Saving memory
del dataset

# Spliting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1, 
                                                    random_state=1969)

# Saving memory
del X, y

# Making the model
model = Sequential()
model.add(Embedding(VOCAB_SIZE, 32, input_length=MAX_WORDS))

model.add(Dropout(0.5))

model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(3))

model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(3))

model.add(Dropout(0.5))

model.add(Bidirectional(LSTM(512, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(128, activation="relu"))
model.add(Dense(32, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(NO_OF_CLASSES, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Printing the model summary
print(model.summary())

monitor = EarlyStopping(monitor='val_loss', 
                        patience=5, 
                        mode='min',
                        restore_best_weights=True)

checkpoint = ModelCheckpoint()

# Starting the training process
model.fit(X_train, 
          y_train, 
          validation_data=(X_test, y_test), 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE, 
          callbacks=[monitor, checkpoint])

# Saving the model
model.save('dataset/model.h5')
