import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
import utils as myutils
import pickle 
import gensim

meta = pd.read_csv('dataset/meta.csv')
meta = meta.sample(frac=1)

# To limit the dataset. Useful for testing
# meta = meta.head(200000)

ids = meta['id'].values
ratings = pd.get_dummies(meta['star']).values

del meta

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('model.bin', binary=True)

X_train, X_test, y_train, y_test = train_test_split(ids, ratings, test_size=0.1, random_state=1969)

del ids, ratings

VOCAB_SIZE = len(word2vec_model.vocab)
MAX_LENGTH = 80
NO_OF_CLASSES = 5
BATCH_SIZE = 64
STEPS_PER_EPOCH = int(len(X_train)/BATCH_SIZE)
VALIDATION_STEPS = int(len(X_test)/BATCH_SIZE)

train_gen = myutils.generate_batch(X=X_train, y=y_train, batch_size=BATCH_SIZE, vocab_length=VOCAB_SIZE, max_length=MAX_LENGTH, no_classes=NO_OF_CLASSES, model=word2vec_model)
test_gen = myutils.generate_batch(X=X_test, y=y_test, batch_size=BATCH_SIZE, vocab_length=VOCAB_SIZE, max_length=MAX_LENGTH, no_classes=NO_OF_CLASSES, model=word2vec_model)

model = Sequential()
model.add(LSTM(32, input_shape=(None, 100*MAX_LENGTH)))
model.add(Dense(NO_OF_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit_generator(train_gen,
                    epochs=2,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=test_gen,
                    validation_steps=VALIDATION_STEPS)

model.save('model_rnn.h5')