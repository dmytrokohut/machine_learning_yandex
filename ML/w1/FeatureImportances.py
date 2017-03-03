# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
data = pd.read_csv('titanic.csv', index_col='PassengerId', usecols=['PassengerId', 'Pclass', 'Fare', 'Age', 'Sex', 'Survived'])

# Обратите внимание, что признак Sex имеет строковые значения.
data['Sex'] = data['Sex'].factorize()[0]

"""В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст.
   Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки."""
data_without_nan = data.dropna(axis=0) 			# axis: 0 - index, 1 - columns (default=0)

# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
feature_names = ['Pclass', 'Fare', 'Age', 'Sex']
new_data = np.array(data_without_nan.as_matrix(columns=feature_names))

# Выделите целевую переменную — она записана в столбце Survived.
target = np.array(data_without_nan.as_matrix(columns=['Survived']).T)[0]

# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).
clf = DecisionTreeClassifier(random_state=241)
clf.fit(new_data, target)

"""Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи
   (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен). """
importances = clf.feature_importances_

feature_importances_dict = dict(zip(feature_names, importances))

feature_importances_dict = Counter(feature_importances_dict)
print "Most important features: '", feature_importances_dict.most_common(1)[0][0], "', '", feature_importances_dict.most_common(2)[1][0], "'"

file = open("task.txt", "w")
result_string = feature_importances_dict.most_common(1)[0][0] + ' ' + feature_importances_dict.most_common(2)[1][0]
file.write(result_string)
file.close()