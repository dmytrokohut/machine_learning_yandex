# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC

"""Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм" (инструкция приведена выше).
   Обратите внимание, что загрузка данных может занять несколько минут"""
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
 
"""Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании мы предлагаем вам вычислить TF-IDF по всем данным. При таком подходе получается,
   что признаки на обучающем множестве используют информацию из тестовой выборки — но такая ситуация вполне законна, поскольку мы не используем значения целевой переменной из теста.
   На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки известны на момент обучения, и поэтому можно ими пользоваться при обучении алгоритма."""
vectorizer = TfidfVectorizer()
newsgroups_train = vectorizer.fit_transform(newsgroups.data)

"""Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром (kernel='linear') при помощи кросс-валидации по 5 блокам.
   Укажите параметр random_state=241 и для SVM, и для KFold. В качестве меры качества используйте долю верных ответов (accuracy)."""
grid = {'C': np.power(10.0, np.arange(-5, 6))}
kf = KFold(n_splits=5, shuffle=True, random_state=241)
svm = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(svm, grid, scoring='accuracy', cv=kf)
gs.fit(newsgroups_train, newsgroups.target)

best_C = gs.get_params()["estimator__C"]

# Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.
clf = SVC(C=best_C, kernel='linear', random_state=241)
clf.fit(newsgroups_train, newsgroups.target)

"""Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле coef_ у svm.SVC). Они являются ответом на это задание.
   Укажите эти слова через запятую или пробел, в нижнем регистре, в лексикографическом порядке."""
most_important_words_indexes = np.argsort(abs(clf.coef_.toarray()[0]))[-10:]

most_important_words = np.array(vectorizer.get_feature_names())[most_important_words_indexes]

most_important_words_sorted = sorted(most_important_words)
result_string = " ".join(most_important_words_sorted)
print result_string

file_answer = open("answer_text.txt", "w")
file_answer.write(result_string)
file_answer.close()
