import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, normalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import backend as K
import random
from keras.utils import to_categorical
#from sklearn.metrics import log_loss


class MyNeuralNets(object):
    def __init__(self, input_dim, n_classes, hyperparams={}, model_setup={}):
        self.model = Sequential()
        self.model_setup = model_setup
        self.input_dim = input_dim
        self.n_classes = n_classes
        #Build defaults, then replace if given
        self.hyperparams = {'epochs': 5, 'batch_size': 32}
        for key in hyperparams:
            self.hyperparams[key] = hyperparams[key]

    def build_model(self, architecture_type='dynamic'):
        if architecture_type == 'simple':
            self.simple_architecture()
        elif architecture_type == 'simple_2':
            self.simple_architecture_2()
        elif architecture_type == 'simple_regularized':
            self.simple_architecture_regularized()
        elif architecture_type == 'dynamic':
            if len(self.model_setup) > 0:
                self.dynamic_architecture(self.model_setup)
            else:
                self.dynamic_architecture()
        else:
            print('No architecture found.')
            assert(0 == 1)

    def dynamic_architecture(self,
                            model_setup={'loss': 'categorical_crossentropy',
                                        'optimizer': 'sgd',
                                        'dense_layers': [100, 64],
                                        'activation': ['relu', 'relu'],
                                        'dropout': [0.5, 0.2],
                                        'regularizer': [regularizers.l2(0.05), regularizers.l2(0.05)],
                                        'batchnormalization': [True, True]}):
        self.model = Sequential()
        for i, units in enumerate(model_setup['dense_layers']):
            if i == 0:
                self.model.add(Dense(units=units, input_dim=self.input_dim, kernel_regularizer=model_setup['regularizer'][i]))
            else:
                self.model.add(Dense(units=units, kernel_regularizer=model_setup['regularizer'][i]))
            if model_setup['batchnormalization']:
                self.model.add(normalization.BatchNormalization())
            self.model.add(Activation(model_setup['activation'][i]))
            self.model.add(Dropout(model_setup['dropout'][i]))

        self.model.add(Dense(units=self.n_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=model_setup['loss'],
                            optimizer=model_setup['optimizer'],
                            metrics=['accuracy'])


    def simple_architecture(self):
        self.model = Sequential()
        self.model.add(Dense(units=100, input_dim=self.input_dim))
        self.model.add(Activation('relu'))
        self.model.add(Dense(units=64))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.n_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])

    def simple_architecture_2(self):
        self.model = Sequential()
        self.model.add(Dense(units=100, input_dim=self.input_dim))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.n_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])

    def simple_architecture_regularized(self):
        self.model = Sequential()
        self.model.add(Dense(units=150, input_dim=self.input_dim,
                            kernel_regularizer=regularizers.l2(0.05)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=100, kernel_regularizer=regularizers.l2(0.1)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=25, kernel_regularizer=regularizers.l2(0.1)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.05))
        self.model.add(Dense(self.n_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])

    def simple_architecture_regularized_org(self):
        self.model = Sequential()
        self.model.add(Dense(units=1500, input_dim=self.input_dim,
                            kernel_regularizer=regularizers.l2(0.05)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=1000, kernel_regularizer=regularizers.l2(0.05)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=254, kernel_regularizer=regularizers.l2(0.05)))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(self.n_classes))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])

    def fit(self, train_data, train_labels):
        if train_labels.shape[1] < 2:
            train_labels = self.get_categorical_labels(train_labels)
        self.model.fit(np.array(train_data), np.array(train_labels), **self.hyperparams)
        return self

    def predict(self, test_data):
        return self.model.predict(np.array(test_data), batch_size=128)

    def fit_predict(self, train_data, train_labels):
        self.model.fit(np.array(train_data), np.array(train_labels), **self.hyperparams)
        return self.predict(np.array(train_data))

    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels, batch_size=128)

    def normalize_labels(self, labels):
        return labels - labels.min()

    def get_categorical_labels(self, labels):
        labels = np.array(labels).reshape((-1, 1))
        if labels.min() > 0:
            labels = self.normalize_labels(labels)
        labels = to_categorical(labels, num_classes=self.n_classes)
        return labels

    def cross_validate(self, train_data, train_labels, cv=5, architecture_type='simple_regularized', seed=12):
        train_data = np.array(train_data)
        train_labels = self.get_categorical_labels(train_labels)
        np.random.seed(seed)
        random.seed(seed)
        indexes = [i for i in range(train_data.shape[0])]
        random.shuffle(indexes)
        n = int(np.floor(train_data.shape[0]/cv))
        scores = np.zeros(cv)
        cur_indexes_test = indexes[:n]
        cur_indexes_train = indexes[n:]
        cur_data_test = train_data[cur_indexes_test, :]
        cur_labels_test = train_labels[cur_indexes_test, :]
        cur_data_train = train_data[cur_indexes_train, :]
        cur_labels_train = train_labels[cur_indexes_train, :]
        self.build_model(architecture_type)
        self.fit(cur_data_train, cur_labels_train)
        scores[0] = self.evaluate(cur_data_test, cur_labels_test)[0]
        for i in range(1, cv-1):
            cur_indexes_test = indexes[(i*n):(i+1)*n]
            cur_indexes_train = indexes[:n*i] + indexes[(i+1)*n:]
            cur_data_test = train_data[cur_indexes_test, :]
            cur_labels_test = train_labels[cur_indexes_test, :]
            cur_data_train = train_data[cur_indexes_train, :]
            cur_labels_train = train_labels[cur_indexes_train, :]
            self.build_model(architecture_type)
            self.fit(cur_data_train, cur_labels_train)
            scores[i] = self.evaluate(cur_data_test, cur_labels_test)[0]
        cur_indexes_test = indexes[(cv-1)*n:]
        cur_indexes_train = indexes[:(cv-1)*n]
        cur_data_test = train_data[cur_indexes_test, :]
        cur_labels_test = train_labels[cur_indexes_test, :]
        cur_data_train = train_data[cur_indexes_train, :]
        cur_labels_train = train_labels[cur_indexes_train, :]
        self.build_model(architecture_type)
        self.fit(cur_data_train, cur_labels_train)
        scores[-1] = self.evaluate(cur_data_test, cur_labels_test)[0]
        return scores

    def multi_class_log_loss(self, labels_true, labels_pred):
        const = np.divide(-1, labels_true.shape[0])
        return const*K.diag(labels_true.dot(K.log(labels_pred.T)))
