# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

"""Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston(). 
   Результатом вызова данной функции является объект, у которого признаки записаны в поле data, а целевой вектор — в поле target."""
data = load_boston()['data']
targets = load_boston()['target']

# Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.
scale_data = scale(data)

"""Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом, чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace).
   Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей. 
   В качестве метрики качества используйте среднеквадратичную ошибку (параметр scoring='mean_squared_error' у cross_val_score;
   при использовании библиотеки scikit-learn версии 18.0.1 и выше необходимо указывать scoring='neg_mean_squared_error').
   Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42, не забудьте включить перемешивание выборки (shuffle=True)."""
p_values = np.linspace(1, 10, num=200)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_accuracy = [cross_val_score(estimator=KNeighborsRegressor(n_neighbors=5, weights='distance', p=p_i, metric='minkowski'), X=scale_data, y=targets, cv=kf).mean() for p_i in p_values]

"""Определите, при каком p качество на кросс-валидации оказалось оптимальным. Обратите внимание, что cross_val_score возвращает массив показателей качества по блокам;
   необходимо максимизировать среднее этих показателей. Это значение параметра и будет ответом на задачу."""
best = p_values[int(max(cv_accuracy))]

print round(best, 2)

file_answer = open("answer.txt", "w")
file_answer.write(repr(round(best, 2)))
file_answer.close()