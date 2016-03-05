# coding: utf8
"""
'''
Найти четырёхзначное число, являющееся полным квадратом, первая цифра которого
равна второй, а третья - четвёртой.
'''
import math as m

for x in range(0, 10):
    for y in range(0, 10):
        if m.sqrt(1000*x+100*x+10*y+y) - int(m.sqrt(1000*x+100*x+10*y+y)) == 0:
            a = 1000*x+100*x+10*y+y
        else:
            b = 1000*x+100*x+10*y+y
        
print "This number is ",a, ". Sqrt(number) = ", int(m.sqrt(a)) 
print b

A = m.cos(2*m.pi/5) + m.cos(4*m.pi/5) # = -1/2
print A


def fact(n):
    if n < 0 or n - int(n) <> 0 :
        return "Sorry, guy, it's not OK"
    elif n == 0:
        return 1
    return fact(n-1)*n

for x in range(0,100):
    if fact(x)/10.0**x < 1:
        a = x
    else:
        b = x

print a
print fact(24.0)/10**(24)
print fact(25.0)/10**(25)






'''
Число цифр в числе
'''
n=2**100
print len(str(n))
"""



"""

'''
Задача.

Все целые числа, начиная с единицы, выписаны подряд.
Таким образом, получается следующий ряд цифр:
123456789101112131415... .
Определить, какая цифра стоит на 206788-м месте.
'''

# (Рецепт от Вовы)

n = 206788

a = map(int,list("".join(map(str,range(n)))))

print(a[n])


# Пояснения

m=10
print range(m) # выведет [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

'''
В Питоне можно вывести список строк при помощи однострочной
команды. Для этого используется метод строки join. У этого
метода один параметр: список строк. В результате возвращается
строка, полученная соединением элементов переданного списка в
одну строку, при этом между элементами списка вставляется разделитель,
равный той строке, к которой применяется метод.
'''

a = ['red', 'green', 'blue']
print(' '.join(a))
# вернёт red green blue
print(''.join(a))
# вернёт redgreenblue
print('***'.join(a))
# вернёт red***green***blue

'''
Если же список состоит из чисел, то придется использовать еще
тёмную магию генераторов.
Вывести элементы списка чисел, разделяя их пробелами, можно так:
'''

a = [1, 2, 3]
print(' '.join([str(i) for i in a]))
# следующая строка, к сожалению, вызывает ошибку:
# print(' '.join(a))


print list('spisok')
# ['s', 'p', 'i', 's', 'o', 'k']

'''
Map

Принимает функцию и набор данных. Создаёт новую коллекцию, выполняет функцию
на каждой позиции данных и добавляет возвращаемое значение в новую коллекцию.
Возвращает новую коллекцию.

Простой map, принимающий список имён и возвращающий список длин:
'''
name_lengths = map(len, ['Mari', 'Peter', 'Oli'])

print name_lengths
# => [4, 5, 3]

# Этот map возводит в квадрат каждый элемент:

squares = map(lambda x: x * x, [0, 1, 2, 3, 4])

print squares
# => [0, 1, 4, 9, 16]

'''
Он не принимает именованную функцию, а берёт анонимную, определённую через
lambda. Параметры lambda определены слева от двоеточия. Тело функции – справа.
Результат возвращается неявным образом.
'''

"""

"""
#Графики в стиле комиксов xkcd


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pylab import *

'''
print np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(
    np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.pi)))))))))))))))))
print np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(
    np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.sqrt(np.e)))))))))))))))))
'''


x = linspace(0.001, 16)
y = x - np.e*np.log(x)

# !!! Включить оформление в стиле xkcd.com
plt.xkcd()

plt.plot (x, y)
csfont = {'fontname':'Comic Sans MS'}
#hfont = {'fontname':'Helvetica'}

plt.title('e^pi > pi^e',**csfont)
plt.xlabel('x label', **csfont)
#plt.xlabel('x')    # обозначение оси абсцисс
plt.ylabel('y = x - eln(x)')    # обозначение оси ординат
#plt.grid(True)

plt.show()



import pylab
import numpy
from mpl_toolkits.mplot3d import Axes3D

def func1 (x):
    return 2 * x + 1


pylab.xkcd()

# Первый график
pylab.subplot (2, 2, 1)

x1 = numpy.arange (0, 10, 0.05)
y1 = func1 (x1)
pylab.plot (x1, y1)


# Второй график
fig = pylab.subplot (2, 2, 2)
# Данные для построения графика
data = [20.0, 10.0, 5.0, 1.0, 0.5]

# Нарисовать круговой график
pylab.pie (data)


# Нарисовать третий график
pylab.subplot (2, 1, 2)
xbar = [1, 2, 3, 4, 5]
hbar = [10.0, 15.0, 5.0, 1.0, 5.5]
pylab.bar (xbar, hbar)

pylab.show()



'''
Как видите, однажды вызвав функцию pylab.xkcd(), эффект начинает применяться
ко всем последующим графикам.
Но если вы хотите применить этот эффект только к одному графику, то можно
воспользоваться оператором with. Дело в том, что функция pylab.xkcd()
возвращает объект, у которого есть метод '_exit_', вызываемый при выходе
из оператора with.

Далее показано, как нужно изменить предыдущий пример, чтобы к круговой
диаграмме не применялся эффект xkcd. 
'''

import pylab
import numpy
from mpl_toolkits.mplot3d import Axes3D

def func1 (x):
    return 2 * x + 1


# Первый график
with pylab.xkcd():
    pylab.subplot (2, 2, 1)

    x1 = numpy.arange (0, 10, 0.05)
    y1 = func1 (x1)
    pylab.plot (x1, y1)


# Второй график
fig = pylab.subplot (2, 2, 2)
# Данные для построения графика
data = [20.0, 10.0, 5.0, 1.0, 0.5]

# Нарисовать круговой график
pylab.pie (data)


# Нарисовать третий график
with pylab.xkcd():
    pylab.subplot (2, 1, 2)
    xbar = [1, 2, 3, 4, 5]
    hbar = [10.0, 15.0, 5.0, 1.0, 5.5]
    pylab.bar (xbar, hbar)

pylab.show()


'''
До сих пор мы использовали функцию pylab.xkcd() без параметров, точнее
с параметрами по умолчанию. Но у этой функции есть три необязательных
параметра, которые позволяют настроить искажения, имитирующие дрожание руки.
Все три параметра являются дробными числами. Объявление функции выглядит
следующим образом:

matplotlib.pyplot.xkcd(scale=1, length=100, randomness=2)

Здесь:

    scale - задает размах дрожания "руки" при рисовании.
    length - задает период дрожания руки. Чем меньше это число, тем больше
            периодов колебания будет на графике.
    randomness - задает величину случайного шума, количество резких дрожаний. 
'''
        


# Трехмерный график

# Построение осей
import pylab
from mpl_toolkits.mplot3d import Axes3D

fig = pylab.figure()
Axes3D(fig)

pylab.show()


# Построение графика функции


import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy

def makeData ():
    x = numpy.arange (-10, 10, 0.1)
    y = numpy.arange (-10, 10, 0.1)
    xgrid, ygrid = numpy.meshgrid(x, y)

    zgrid = numpy.sin (xgrid) * numpy.sin (ygrid) / (xgrid * ygrid)
    return xgrid, ygrid, zgrid

pylab.xkcd(scale=6, length=100, randomness=4)

x, y, z = makeData()

fig = pylab.figure()
axes = Axes3D(fig)

axes.plot_surface(x, y, z, color='#11aa55')

pylab.show()

"""


# Файлы

"""
f = open('data.txt', 'w') # Создается новый файл для вывода
f.write('Hello\n')  # Запись строки байтов в файл

f.write('world\n')  

f.close()  # Закрывает файл и выталкивает выходные буферы на диск


f = open('data.txt') # ‘r’ – это режим доступа к файлу по умолчанию

text = f.read() # Файл читается целиком в строку
 
print(text)  # Вывод, с попутной интерпретацией служебных символов

print text.split() # Содержимое файла всегда является строкой
"""

'''
Часто используемые операции над файлами


output = open(r’C:\spam’, ‘w’) - Открывает файл для записи  (‘w’ означает write – запись)

input = open(‘data’, ‘r’) - Открывает файл для чтения (‘r’ означает read – чтение)

input = open(‘data’) - То же самое, что и в предыдущей строке  (режим ‘r’ используется по умолчанию)

aString = input.read() - Чтение файла целиком в единственную строку

aString = input.read(N) - Чтение следующих N символов (или байтов) в строку

aString = input.readline() - Чтение следующей текстовой строки (включая символ конца строки) в строку

aList = input.readlines() - Чтение файла целиком в список строк (включая символ конца строки)

output.write(aString) - Запись строки символов (или байтов) в файл

output.writelines(aList) - Запись всех строк из списка в файл

output.close() - Закрытие файла вручную (выполняется по окончании работы с файлом)

output.flush() - Выталкивает выходные буферы на диск, файл остается открытым

anyFile.seek(N) - Изменяет текущую позицию в файле для следующей операции, смещая ее на N байтов от начала файла.

for line in open(‘data’): - Итерации по файлу, построчное чтение
'''

"""
myfile = open('myfile.txt', 'w')
myfile.write('hello text file\n')
myfile.close()

myfile = open('myfile.txt')
print(open('myfile.txt').read())


X, Y, Z = 43, 44, 45          # Объекты языка Python должны
S = 'Spam'                    # записываться в файл только в виде строк
D = {'a': 1, 'b': 2}
L = [1, 2, 3]

F = open('datafile.txt', 'w') 
F.write(S + '\n')             # Строки завершаются символом \n
F.write('%s,%s,%s\n' % (X, Y, Z))     # Преобразует числа в строки
F.write(str(L) + '$' + str(D) + '\n') # Преобразует и разделяет символом $
F.close()


chars = open('datafile.txt').read() # Отображение строки

print(chars)

'''
Теперь нам необходимо выполнить обратные преобразования, чтобы получить 
из строк в текстовом файле действительные объекты языка Python. Интер-
претатор Python никогда автоматически не выполняет преобразование строк 
в числа или в объекты других типов, поэтому нам необходимо выполнить соот-
ветствующие преобразования, чтобы можно было использовать операции над 
этими объектами, такие как индексирование, сложение и так далее:
'''

F = open('datafile.txt') # Открыть файл снова
line = F.readline()      # Прочитать одну строку
line.rstrip()            # Удалить символ конца строки

line = F.readline()        # Следующая строка из файла
# Это - строка ‘43,44,45\n’
parts = line.split(',')    # Разбить на подстроки по запятым
print parts

numbers = [int(P) for P in parts] # Преобразовать весь список в писок
                                    # целых чисел
print numbers

'''
для удаления завер-
шающего символа \n в конце последней подстроки не был использован метод 
rstrip, потому что int и некоторые другие функции преобразования просто иг-
норируют символы-разделители, окружающие цифры.

чтобы преобразовать список и словарь в третьей строке файла, мож-
но воспользоваться встроенной функцией eval, которая интерпретирует строку 
как программный код на языке Python (формально – строку, содержащую вы-
ражение на языке Python):
'''

line = F.readline()
print line
parts = line.split('$') # Разбить на строки по символу $
print line

objects = [eval(P) for P in parts] # преобразует все строки из списка в объекты
print objects

"""

# Pickling

'''
засолка, называется так, потому что «сохраняет» структуру 
данных. Модуль pickle содержит необходимые команды. Чтобы его использовать, 
импортируйте  pickle и затем как обычно откройте файл:
Слово pickling имеет следующие значения:1) квашение, засол, маринование;
2) протравливание, травление, декапирование.
'''

import pickle

f = open("test.pck","w")

pickle.dump(12.3, f)
pickle.dump([1,2,3], f)

f.close() 

f = open("test.pck","r")

x = pickle.load(f)
print x, type(x)

y = pickle.load(f)
print y, type(y)










