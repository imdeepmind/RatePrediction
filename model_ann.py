# Importing the dependencies
import pandas as pd
import gensim
from sklearn.model_selection import train_test_split
from utils import generate_batch
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Word2Vec Model
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True)

# Reading the dataset
meta = pd.read_csv('dataset/meta.csv')
meta = meta.sample(frac=1)

# To limit the dataset. Useful for testing
#meta = meta.head(5000)

# Selecting ids and ratings(star) column
ids = meta['id'].values
ratings = pd.get_dummies(meta['star']).values

# Deleting the variable meta
del meta

# Spliting the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(ids, ratings, test_size=0.1, random_state=1969)

# Deleting the ids and ratings variable
del ids, ratings

# Hyperparameters and Constants
VOCAB_SIZE = len(word2vec_model.vocab)
MAX_LENGTH = 80
NO_OF_CLASSES = 5
BATCH_SIZE = 64
STEPS_PER_EPOCH = int(len(X_train)/BATCH_SIZE)
VALIDATION_STEPS = int(len(X_test)/BATCH_SIZE)

# Generator functions
train_gen = generate_batch(X=X_train,
                           y=y_train, 
                           batch_size=BATCH_SIZE, 
                           vocab_length=VOCAB_SIZE, 
                           max_length=MAX_LENGTH, 
                           no_classes=NO_OF_CLASSES, 
                           model=word2vec_model)

test_gen = generate_batch(X=X_test, 
                          y=y_test, 
                          batch_size=BATCH_SIZE, 
                          vocab_length=VOCAB_SIZE, 
                          max_length=MAX_LENGTH, 
                          no_classes=NO_OF_CLASSES, 
                          model=word2vec_model)

# ANN Model
model = Sequential()
model.add(Dense(150, activation='relu', input_dim=8000))
model.add(Dropout(0.50))
model.add(Dense(NO_OF_CLASSES, activation='softmax'))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Printing the summary of the model
print(model.summary())

# Starting to train the model
model.fit_generator(train_gen,
                    epochs=2,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=test_gen,
                    validation_steps=VALIDATION_STEPS)

# Saving the model into a h5 file
model.save('model_ann.h5')