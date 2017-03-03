# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.svm import SVC

# Загрузите выборку из файла svm-data.csv. В нем записана двумерная выборка (целевая переменная указана в первом столбце, признаки — во втором и третьем).
data = pd.read_csv('svm-data.csv', index_col=None, header=None)
classes = data[0]
observations = data.ix[:,1:].copy()

"""Обучите классификатор с линейным ядром, параметром C = 100000 и random_state=241. Такое значение параметра нужно использовать, чтобы убедиться, что SVM работает с выборкой 
   как с линейно разделимой. При более низких значениях параметра алгоритм будет настраиваться с учетом слагаемого в функционале,  штрафующего за маленькие отступы,
   из-за чего результат может не совпасть с решением классической задачи SVM для линейно разделимой выборки."""
svm = SVC(C=100000, kernel='linear', random_state=241)
svm.fit(observations, classes)

"""Найдите номера объектов, которые являются опорными (нумерация с единицы). Они будут являться ответом на задание.
   Обратите внимание, что в качестве ответа нужно привести номера объектов в возрастающем порядке через запятую или пробел. Нумерация начинается с 1."""
support_indexes = svm.support_
str_support_indexes = [repr(i+1) for i in support_indexes]

result_string = " ".join(str_support_indexes)
print result_string

file_answer = open("answer.txt", "w")
file_answer.write(result_string)
file_answer.close()