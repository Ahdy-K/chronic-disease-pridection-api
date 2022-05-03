# -*- coding: utf-8 -*-


import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def heart_disease_predictor():
  heart_data = pd.read_csv('./heart.csv')

  """notre dataset contient 303 lignes et 14 colonnes"""

  our_features = heart_data.drop(columns='target', axis=1)
  our_target = heart_data['target']
  our_features_train, our_features_test, our_target_train, our_target_test = train_test_split(our_features,our_target, test_size=0.2, stratify = our_target, random_state = 3)
  model = LogisticRegression()
  model.fit(our_features_train, our_target_train)
  our_features_prediction = model.predict(our_features_train)
  training_data_accuracy = accuracy_score(our_features_prediction, our_target_train)
  test_prediction = model.predict(our_features_test)
  test_accuracy = accuracy_score(test_prediction, our_target_test)



  """# Création de systeme de prédiction :"""

  donnee= (49,1,1,130,266,0,1,171,0,0.6,2,0,2)
  donnee_array = np.asarray(donnee)
  donnee_traitee = donnee_array.reshape(1,-1)

  prediction = model.predict(donnee_traitee)
  print(prediction)

  if (prediction[0]==0):
    return ('This patient is healthy and there is no risk of heart diseases in the near future')
  else:
    return ('unfortunately there is a big chance of getting heart diseases in the future be healthy')