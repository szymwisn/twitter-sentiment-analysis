import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPooling1D, Conv1D, BatchNormalization, Dropout
from tensorflow.keras import regularizers 


def create_model(optimizer='adam', loss='categorical_crossentropy'):
    # cnn
    embed_dim = 128
    f_conv_size = 256
    f_dense_size = 16
    out_dim = 3

    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length = X.shape[1]))
    model.add(Conv1D(f_conv_size, kernel_size=3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(f_dense_size, kernel_regularizer=regularizers.l2(0.01), activation='tanh'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(out_dim, activation='softmax'))
    model.compile(loss = loss, optimizer=optimizer, metrics = ['accuracy'])
    model.summary()

    return model

def find_cnn_model_params(X_train, X_test, Y_train, Y_test):
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
    
    
    with open("params_sorted_by_mean_cnn_model.txt", "a") as file:
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

#find_cnn_model_params(X_train, X_test, Y_train, Y_test)




epochs_nb = 20
batch_size = 64
model = create_model()

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=batch_size,
                        epochs=epochs_nb)

results = model.evaluate(X_test, Y_test, batch_size=batch_size)
print("test loss, test acc:", results)


with open("cnn_results.txt", "a") as file:
        file.write(str(results))