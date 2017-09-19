import numpy as np
import pandas as pd
from build_net import MyNeuralNets
from keras import regularizers
from keras import optimizers
from sklearn.preprocessing import StandardScaler



def create_submission_normalized(hyperparams, name_string, model_setup={}):
    nn_simple = MyNeuralNets(input_dim=train.shape[1]-1,
                            n_classes=len(np.unique(train['Class'])),
                            hyperparams=hyperparams,
                            model_setup=model_setup)
    nn_simple.build_model()
    nn_simple.fit(train_normalized.iloc[:, :-1], np.array(train_normalized.iloc[:, -1]).reshape((-1, 1)))
    predicted_labels = nn_simple.predict(test_normalized)
    predicted_labels = pd.DataFrame(predicted_labels,
                                    index=test.index,
                                    columns=['class1',
                                    'class2','class3',
                                    'class4','class5',
                                    'class6','class7',
                                    'class8','class9'])
    predicted_labels.to_csv('data/predicted_labels/'+name_string+'.csv')

def cross_validate(hyperparams, cur_model):
    nn_simple = MyNeuralNets(input_dim=train.shape[1]-1,
                            n_classes=len(np.unique(train['Class'])),
                            hyperparams=hyperparams,
                            model_setup=cur_model)
    scores = nn_simple.cross_validate(train.iloc[:, :-1], train.iloc[:, -1], cv=3, architecture_type='dynamic')
    return scores

def cross_validate_normalized(hyperparams, cur_model):
    nn_simple = MyNeuralNets(input_dim=train.shape[1]-1,
                            n_classes=len(np.unique(train['Class'])),
                            hyperparams=hyperparams,
                            model_setup=cur_model)
    scores = nn_simple.cross_validate(train_normalized.iloc[:, :-1], train_normalized.iloc[:, -1], cv=3, architecture_type='dynamic')
    return scores

train = pd.read_csv('data/train_1.csv', index_col=0)
test = pd.read_csv('data/test_1.csv', index_col=0)

scaler = StandardScaler()
train_normalized = train.copy()
scaler.fit(train_normalized.iloc[:, :-1])
train_normalized.iloc[:, :-1] = scaler.transform(train_normalized.iloc[:, :-1])
test_normalized = test.copy()
test_normalized.iloc[:, :] = scaler.transform(test.iloc[:, :])

#hyperparams = {'epochs': 500, 'batch_size': 512}
#create_submission(hyperparams, 'test_1_droput_reg_big_3')
def create_model(learning_rate):
    return {'loss': 'categorical_crossentropy',
            'optimizer': optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
            'dense_layers': [500, 250, 64, 32],
            'activation': ['relu', 'relu', 'relu', 'relu'],
            'dropout': [0.5, 0.3, 0.1, 0],
            'regularizer': [regularizers.l2(0.2), regularizers.l2(0.1), regularizers.l2(0.05), regularizers.l2(0.05)],
            'batchnormalization': [0, 0, 0, 0]}


adam_1 = {'loss': 'categorical_crossentropy',
            'optimizer': optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=10**-3),
            'dense_layers': [124, 64, 32, 16],
            'activation': ['relu', 'relu', 'relu', 'relu'],
            'dropout': [0.5, 0.3, 0.1, 0],
            'regularizer': [regularizers.l2(0.2), regularizers.l2(0.1), regularizers.l2(0.05), regularizers.l2(0.05)],
            'batchnormalization': [1, 1, 0, 0]}


#hyperparams = {'epochs': 300, 'batch_size': 512}
#create_submission(hyperparams, 'test_30_aug_1', adam_2)
#create_submission_normalized(hyperparams, 'test_day_2_normalized', adam_3)

model_list = [adam_1]
model_name_list = ['Adam 1']
epochs_list = [100]
batch_size_list = [32, 512]
#epochs_list = []
#batch_size_list = []
scores_matrix = np.zeros((3, len(epochs_list)*len(batch_size_list)*len(model_list)))
i = 0
for model_n, model in enumerate(model_list):
    for epoch in epochs_list:
        for batch_size in batch_size_list:
            hyperparams = {'epochs': epoch, 'batch_size': batch_size}
            cur_model = model
            scores = cross_validate_normalized(hyperparams, cur_model)
            scores_matrix[:, i] = scores
            i += 1

def print_results(scores_matrix):
    k = 0
    for model_name in model_name_list:
        for i in range(len(epochs_list)):
            for j in range(len(batch_size_list)):
                print('\n',
                    'Epochs:', epochs_list[i], ', ',
                    'Model: {}'.format(model_name), ', ',
                    'Batch size:', batch_size_list[j],',',
                    'Mean:', np.mean(scores_matrix[:, k]),
                    'std:', np.std(scores_matrix[:, k]))
                k += 1

print('Original data')
print_results(scores_matrix)

#create_submission(hyperparams, 'second_submission')
