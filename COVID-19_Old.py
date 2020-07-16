# Importing all essential libraries

import os
import random
import sys
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import LambdaCallback
from keras.layers import Flatten

#-------------------------------------------------------------------#

# Change working directory

# os.chdir(desired_working_directory) 
# print(os.getcwd())

#-------------------------------------------------------------------#

# Reading dataset

dataset = pd.read_csv('dataset_for_training.csv', sep=',')
data = open('dataset.csv', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d, %d unique' % (data_size, vocab_size)) # Identifying all unique characters

#-------------------------------------------------------------------#

# Giving a key for all the uniquqe characters

char_indices = {ch:i for i, ch in enumerate(chars)}
indices_char = {i:ch for ch, i in enumerate(chars)}
print(char_indices)

#-------------------------------------------------------------------#

# Vectorizing inputs to pass it in the model

import numpy as np
maxlen = 60
step = 3
smiles = []
corona_smiles = []
for i in range(0, len(data) - maxlen, step):
    smiles.append(data[i: i + maxlen])
    corona_smiles.append(data[i + maxlen])
print('nb sequences:', len(smiles))

print('Vectorization...')
x = np.zeros((len(smiles), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(smiles), len(chars)), dtype=np.bool)
for i, smiles in enumerate(smiles):
    for t, char in enumerate(smiles):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[corona_smiles[i]]] = 1

#-------------------------------------------------------------------#

# BUILDING THE MODEL

model = Sequential()

model.add(LSTM(units = 128, return_sequences = True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.3))

model.add(LSTM(units = 128, return_sequences = True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.3))

model.add(LSTM(units = 128, return_sequences = True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.summary()

#-------------------------------------------------------------------#

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

#-------------------------------------------------------------------#

model.fit(x, y,
          batch_size=256,
          epochs=100,
          callbacks=[print_callback])

#-------------------------------------------------------------------#

model.save_weights(desired_working_directory)
print('Saved model')

#-------------------------------------------------------------------#

# load weights into new model
# loaded_model.load_weights("COVID_19.hdf5")
# print("Loaded model")
