import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers 




def create_model(optimizer='rmsprop', loss='categorical_crossentropy'):
    embed_dim = 128
    f_lstm_size = 196
    f_dense_size = 256
    out_dim = 3

    # rnn 
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length = (83)))
    model.add(LSTM(f_lstm_size, dropout=0.3, recurrent_dropout=0.3, activation='relu'))
    model.add(Dense(f_dense_size, kernel_regularizer=regularizers.l2(0.01), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(out_dim, activation='softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics = ['accuracy'])
    print(model.summary())

    return model


def find_rnn_model_params(X_train, X_test, Y_train, Y_test):
    model = KerasClassifier(build_fn=create_model)


    param_grid = {
        'epochs': [5, 10, 20],
        'batch_size': [16, 32, 64],
        'optimizer': ['rmsprop', 'adam', 'SGD'],
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        cv=KFold(n_splits=5, random_state=1410, shuffle=True),
                        n_jobs=-1, return_train_score=True)

    grid_result = grid.fit(X_train, Y_train, validation_data=(X_test, Y_test))

    df = pd.DataFrame(grid_result.cv_results_).sort_values('mean_test_score', ascending=False)

    with open("params_sorted_by_mean_rnn_model.txt", "a") as file:
        file.write(df.to_string())
        file.write("\n")



X = np.load('../data-analysis/data/X_rnn.npy', allow_pickle=True)
y = np.load('../data-analysis/data/y_rnn.npy', allow_pickle=True)
y = pd.get_dummies(y) # positive -> [1 0 0], neutral ->[0 1 0] negative -> [0 0 1]


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#find_rnn_model_params(X_train, X_test, Y_train, Y_test)





epochs = 5
batch_size = 32
model = create_model()

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size,
                        epochs=epochs)

results = model.evaluate(X_test, Y_test, batch_size=batch_size)
print("test loss, test acc:", results)

with open("rnn_results.txt", "a") as file:
        file.write(str(results))

