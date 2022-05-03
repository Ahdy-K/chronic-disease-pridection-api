# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score



def diabetes_pridection_model(input_data):
    diabetes_dataset = pd.read_csv('./diabetes.csv')
    diabetes_dataset.head()
    diabetes_dataset.shape
    diabetes_dataset.describe()
    diabetes_dataset['Outcome'].value_counts()
    diabetes_dataset.groupby('Outcome').mean()
    X = diabetes_dataset.drop(columns="Outcome", axis=1)
    Y = diabetes_dataset["Outcome"]

    scaler = StandardScaler()

    scaler.fit(X)

    standarized_data = scaler.transform(X)



    X = standarized_data



    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=10)

    print(X.shape, X_train.shape, X_test.shape)



    classifier = svm.SVC(kernel='linear')

    # training
    classifier.fit(X_train, Y_train)



    # accuracy score of the training data:
    X_train_prediction = classifier.predict(X_train)
    model_accuracy = accuracy_score(X_train_prediction, Y_train)

    print("Model accuracy on training data is :", model_accuracy)

    """accuracy score : test data"""


    X_test_prediction = classifier.predict(X_test)
    model_accuracy_test = accuracy_score(X_test_prediction, Y_test)

    print("Model accuracy on test data is :", model_accuracy_test)

    """# Diabetes Prediction System:"""


    input_array = np.asarray(input_data)

    input_data_reshaped = input_array.reshape(1, -1)


    std_data = scaler.transform(input_data_reshaped)
    print(std_data)

    prediction = classifier.predict(std_data)

    if prediction[0] == 1:

        return "unfortunately this person is in risk of getting diabetes in the future"
    else:

        return "This person is healthy and there is no signals of diabetes in the near future "

