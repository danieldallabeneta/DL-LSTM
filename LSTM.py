import numpy as np
import itertools
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix, accuracy_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt

# Read the CSV file
dataset = 'exec'
data = pd.read_csv(dataset + ".csv")
model1 = []
model2 = []
model1.append(data.drop(columns=['change']))
model1.append(data['change'])
model2.append(data.drop(columns=['change', 'type', 'severity', 'resolution', 'status', 'effort']))
model2.append(data['change'])
models = [{'key': 'Complete', 'value': model1}, {'key': 'NotComplete', 'value': model2}]

def get_scores(y_test, y_pred, dataset, algorithm, model):
    scores = []
    scores.append(dataset)
    scores.append(algorithm)
    scores.append(model)

    # F1-Score
    scores.append(f1_score(y_test, y_pred, average='micro'))
    print("F1-Score(micro): " + str(scores[-1]))
    scores.append(f1_score(y_test, y_pred, average='macro'))
    print("F1-Score(macro): " + str(scores[-1]))
    scores.append(f1_score(y_test, y_pred, average='weighted'))
    print("F1-Score(weighted): " + str(scores[-1]))
    scores.append(f1_score(y_test, y_pred, average=None))
    print("F1-Score(None): " + str(scores[-1]))

    #Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Accuracy
    scores.append(accuracy_score(y_test, y_pred, normalize=True))
    print("Accuracy: " + str(scores[-1]))

    # precision
    precision = tp/ (tp + fp)
    scores.append(precision)

    # recall
    recall = tp / (tp + fn)
    scores.append(recall)
    print("Recall: " + str(scores[-1]))

    # Specificity
    specificity = tn / (tn + fp)
    scores.append(specificity)
    print("Specificity: " + str(scores[-1]))

    # Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix: [" + str(cnf_matrix[0][0]) + ", " + str(round(cnf_matrix[1][1], 2)) + "]")
    plot_confusion_matrix(cnf_matrix, dataset, algorithm)

    # ROC_AUC
    scores.append(roc_auc_score(y_test, y_pred))
    print("ROC AUC score: " + str(scores[-1]))

    scores.append([tn, fp, fn, tp])

    head = ['Dataset', 'Algoritm', 'model', 'F1-Score(micro)', 'F1-Score(macro)',
            'F1-Score(weighted)', 'F1-Score(None)', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'ROC AUC score',
            'Confusion matrix']

    if not os.path.exists('results/' + dataset + '.csv'):
        f = open("results/" + dataset + ".csv", "a")
        writer = csv.writer(f)
        writer.writerow(head)
        f.close()

    f = open("results/" + dataset + ".csv", "a")
    writer = csv.writer(f)
    writer.writerow(scores)
    f.close()

    return scores

def plot_confusion_matrix(cm, dataset, algorithm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()    

    fmt = '.2f' if normalize else 'f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('results/cf-' + dataset + '-' + algorithm +'.png')
    plt.close()

# Define create_model function for KerasClassifier
def create_model(units=50, dropout_rate=0.2, optimizer='adam', learning_rate=0.01, momentum=0.0):   
    if optimizer == 'sgd':
        opt = SGD(learning_rate, momentum)
    else:
        opt = Adam(learning_rate)

    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

for modelo in models:
    X = modelo.get('value')[0]
    y = modelo.get('value')[1]

    # Encode categorical labels if necessary
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Create KerasClassifier wrapper for GridSearchCV
    model = KerasClassifier(model=create_model, verbose=0)

    # Define hyperparameters to search
    param_grid = {
        'model__optimizer': ['adam', 'rmsprop'],
        'model__dropout_rate': [0.1, 0.2, 0.3],
        'model__units': [50, 100, 150],
        'model__learning_rate': [0.01, 0.1],
        'model__momentum': [0.5, 0.9],
        'epochs': [10,20,30],
        'batch_size': [32, 64, 128]  
    }

    # Apply stratified k-fold cross-validation on train set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf, n_jobs=-1)
    grid_result = grid_search.fit(X_train, y_train)

    # Print best hyperparameters
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Evaluate the best model on the test set
    best_model = grid_result.best_estimator_
    #y_pred = best_model.predict(X_test.values.reshape(-1, X_test.shape[1], 1))
    y_pred = best_model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    get_scores(y_test, y_pred, dataset, "LSTM", modelo.get('key'))

    # Calculate accuracy
    """ accuracy = accuracy_score(y_test, y_pred_binary)
    print("Test Accuracy:", accuracy) """