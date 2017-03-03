# -*- coding: utf-8 -*-

from collections import Counter
import pandas as pd
data = pd.read_csv('titanic.csv', index_col='PassengerId')

#Task 1: Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
file_one = open("task_1.txt", "w")

sex_all = data['Sex'].value_counts()
men = sex_all['male']
womans = sex_all['female']
print "Men: ", men, "\tWomans: ", womans
result_string = repr(men) + ' ' + repr(womans)

file_one.write(result_string)
file_one.close()

"""Task 2: Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров. Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен),
 округлив до двух знаков."""
file_two = open("task_2.txt", "w")

all_passengers = data['Survived'].value_counts()
number_of_survived = all_passengers[1]
number_of_death = all_passengers[0]
percent_of_survived = (float)(number_of_survived) / (float)(number_of_survived + number_of_death) * 100
result_string = round(percent_of_survived, 2)
print "Percent of survived: ", result_string, "%"
file_two.write(repr(result_string))
file_two.close()

"""Task 3:Какую долю пассажиры первого класса составляли среди всех пассажиров? Ответ приведите в процентах 
	(число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков."""
file_three = open("task_3.txt", "w")

first_class_passengers = data['Pclass'].value_counts()[1]
all_passengers = data['Pclass'].count()
percent_of_first_class = (float)(first_class_passengers) / (float)(all_passengers) * 100
result_string = round(percent_of_first_class, 2)
print "Percent of first class passengers: ", result_string, "%"

file_three.write(repr(result_string))
file_three.close()

#Task 4: Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. В качестве ответа приведите два числа через пробел.
file_four = open("task_4.txt", "w")

age_mean = data['Age'].mean()
age_median = data['Age'].median()
print "Average mean: ", round(age_mean, 2), "\tMedian: ", age_median
result_string = repr(round(age_mean, 2)) + ' ' + repr(age_median)

file_four.write(result_string)
file_four.close()

#Task 5: Коррелируют ли число братьев/сестер/супругов с числом родителей/детей? Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
file_five = open("task_5.txt", "w")

corr = data['SibSp'].corr(data['Parch'], method='pearson')
result_string = round(corr, 2)
print "Correlation beetwen 'SibPh' and 'Parch': ", result_string

file_five.write(repr(result_string))
file_five.close()

"""Task 6: Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name). 
 Это задание — типичный пример того, с чем сталкивается специалист по анализу данных. Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию.
 Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен, а также разделения их на женские и мужские."""

file_six = open("task_6.txt", "w")

#Choose the names of all womans in separate DataFrame
womans_names = data.ix[data['Sex'] == 'female']['Name']

#For unmarried ladies want to drop the name and Miss treatment. (looking for a regular expression)
womans_names = womans_names.str.replace(r'[^,]*, Miss\. ', '')

#Ladies we are interested in their maiden name to married, to be written in brackets. (First remove all the brackets to, and then everything after the closing parenthesis)
womans_names = womans_names.str.replace(r'[^\(]*\(', '')
womans_names = womans_names.str.replace(r'\).*', '')

#Then you have to break all the words of this vector using the space bar. And count the number.
womans_names = [j for i in womans_names.tolist() for j in i.split(' ')]

number_of_names = Counter(womans_names)
result_string = number_of_names.most_common(1)[0][0]

print "Most common name is '", result_string, "'"

file_six.write(result_string)
file_six.close()