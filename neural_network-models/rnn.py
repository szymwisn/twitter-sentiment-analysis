import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GlobalMaxPooling1D, Conv1D, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical



X = np.load('../data-analysis/data/X_rnn.npy', allow_pickle=True)
y = np.load('../data-analysis/data/y_rnn.npy', allow_pickle=True)
y = pd.get_dummies(y) # positive -> [1 0 0], neutral ->[0 1 0] negative -> [0 0 1]


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X)


X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)



embed_dim = 128
lstm_out = 196

# rnn 
model = Sequential()
model.add(Embedding(max_fatures, embed_dim, input_length = X.shape[1]))
model.add(LSTM(lstm_out, dropout=0.2, return_sequences=True))
model.add(LSTM(lstm_out, dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())


epochs = 10
batch_size = 32

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size,
                        epochs=epochs)

validation_size = 1000

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
results = model.evaluate(X_test, Y_test, batch_size=batch_size)
print("test loss, test acc:", results)
