import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Note: Use LSTM if you dont have a NVDIA  else use CuDNNLSTM
from keras.layers import LSTM, CuDNNLSTM, Dense, Dropout, GlobalMaxPool1D, Bidirectional, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Reading the tokenizer pickle file
with open(r"dataset/word_tokenizer.pickle", "rb") as input_file:
    tokenizer = pickle.load(input_file)

# Constants for model configuration
GPU = False
MAX_WORDS = 80
NO_OF_CLASSES = 5
VOCAB_SIZE = 10000
EPOCHS = 50
BATCH_SIZE = 1024

# Loading the dataset
dataset = np.load('dataset/preprocessed_dataset.npy')

# For testing select a fraction of the dataset
# dataset = dataset[0:1000, :]

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
model.add(Embedding(VOCAB_SIZE, 128, input_length=MAX_WORDS))

model.add(Dropout(0.5))

model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(3))

model.add(Conv1D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(3))

model.add(Dropout(0.5))

if GPU:
    model.add(Bidirectional(CuDNNLSTM(256, return_sequences = True)))
else:
    model.add(Bidirectional(LSTM(256, return_sequences = True)))
    
model.add(GlobalMaxPool1D())

model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(NO_OF_CLASSES, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Printing the model summary
print(model.summary())

# Added early stopping system to monnitor validation loss on each epoch and stops training when validation loss start to increase
monitor = EarlyStopping(monitor='val_loss', 
                        patience=5, 
                        mode='min',
                        restore_best_weights=True)

# Saving the model in every epochs for some experiments
checkpoint = ModelCheckpoint(filepath="weights/model.{epoch:02d}-{val_loss:.2f}.h5")

# Starting the training process
model.fit(X_train, 
          y_train, 
          validation_data=(X_test, y_test), 
          epochs=EPOCHS, 
          batch_size=BATCH_SIZE, 
          callbacks=[monitor, checkpoint])

# Saving the model
model.save('weights/model_best.h5')
