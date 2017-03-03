# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv. Целевая переменная записана в первом столбце, признаки — во втором и третьем.
train_data = pd.read_csv('perceptron-train.csv', index_col=None, header=None)
test_data = pd.read_csv('perceptron-test.csv', index_col=None, header=None)

train_classes = train_data[0]
train_observations = train_data.ix[:,1:].copy()
test_classes = test_data[0]
test_observations = test_data.ix[:,1:].copy()

# Обучите персептрон со стандартными параметрами и random_state=241.
clf = Perceptron(random_state=241)
clf.fit(train_observations, train_classes)
test_classes_predicted_no_scaling = clf.predict(test_observations)

# Подсчитайте качество (долю правильно классифицированных объектов, accuracy) полученного классификатора на тестовой выборке.
accuracy_no_scaling = accuracy_score(test_classes, test_classes_predicted_no_scaling, normalize=True)

# Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
scaler = StandardScaler()
train_observations_scaled = scaler.fit_transform(train_observations)
test_observations_scaled = scaler.transform(test_observations)

# Обучите персептрон на новых выборках. Найдите долю правильных ответов на тестовой выборке.
clf.fit(train_observations_scaled, train_classes)
test_classes_predicted_scaling = clf.predict(test_observations_scaled)

accuracy_scaling = accuracy_score(test_classes, test_classes_predicted_scaling, normalize=True)

# Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее. Это число и будет ответом на задание.
result_string = accuracy_scaling - accuracy_no_scaling
print result_string

file_answer = open("answer_normalize.txt", "w")
file_answer.write(repr(round(result_string, 3)))
file_answer.close()