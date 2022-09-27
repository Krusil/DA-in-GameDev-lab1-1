# DA-in-GameDev-lab1
# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #1 выполнил(а):
- Нечаев Никита Вадимович
- РИ-210934
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с основными операторами зыка Python на примере реализации линейной регрессии.

## Задание 1
### Написать программы Hello World на Python и Unity.
Ход работы:
Вывод на Python в Google.colab:
![image](https://user-images.githubusercontent.com/114502827/192612027-747fc140-035e-4fc5-ab5b-57fbd98da5d3.png)
И сохранение на Google Диск:
![image](https://user-images.githubusercontent.com/114502827/192612234-c6bca72e-d29a-4f12-907b-35ca3b0d8699.png)

Вывод на Unity:
![image](https://user-images.githubusercontent.com/114502827/192612857-329b7b2f-850a-4704-80f5-7151064f4020.png)
![image](https://user-images.githubusercontent.com/114502827/192612881-4a71afd7-b9cb-4285-bb97-dbc972858385.png)



## Задание 2
### В разделе «ход работы» пошагово выполнить каждый пункт с описанием и примером реализации задачи по теме лабораторной работы.
Ход работы:
1) Произвести подготовку данных для работы с алгоритмом линейной регрессии. 10 видов данных были установлены случайным образом, и
данные находились в линейной зависимости. Данные преобразуются вформат массива, чтобы их можно было вычислить напрямую при использовании 
умножения и сложения.
![image](https://user-images.githubusercontent.com/114502827/192621966-efd7dbc2-3b2e-49da-bb39-f23a6a2da918.png)

2) Определите связанные функции. Функция модели: определяет модель линейной регрессии wx+b. Функция потерь: функция потерь
среднеквадратичной ошибки. Функция оптимизации: градиентного спуска для нахождения частных производных w и b.
		
```py
def model(a, b, x):
    return a * x + b


def loss_function(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    return (0.5/num) * (np.square(prediction - y)).sum()


def optimize(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    da = (1.0/num) * ((prediction - y)*x).sum()
    db = (1.0/num) * ((prediction - y).sum())
    a = a - Lr * da
    b = b - Lr * db
    return a, b


def iterate(a, b, x, y, times):
    for i in range(times):
	a, b = optimize(a, b, x, y)
	return a, b
	
```

3) Начать итерацию:
 Шаг 1 Инициализация и модель итеративной оптимизации
 ```py
 import numpy as np
import matplotlib.pyplot as plt

x = [3, 21, 22, 34, 54, 34, 55, 67, 89, 99]
x = np.array(x)
y = [2, 22, 24, 65, 79, 82, 55, 130, 150, 199]
y = np.array(y)

def model(a, b, x):
    return a * x + b

def loss_function(a, b, x, y):
    num = len(x)
    prediction = model(a, b, x)
    return (0.5 / num) * (np.square(prediction - y)).sum()

def optimize(a, b, x, y, Lr):
    num = len(x)
    prediction = model(a, b, x)
    da = (1.0 / num) * ((prediction - y) * x).sum()
    db = (1.0 / num) * ((prediction - y).sum())
    a = a - Lr * da
    b = b - Lr * db
    return a, b

def iterate(a, b, x, y, times):
    for i in range(times):
        a, b = optimize(a, b, x, y, Lr)
    return a,b

a = np.random.rand(1)
print(a)
b = np.random.rand(1)
print(b)
Lr = 0.000001

a, b = iterate(a, b, x, y, 1)
prediction = model(a, b, x)
loss = loss_function(a, b, x, y)
print(a, b, loss)
plt.scatter(x, y)
plt.plot(x, prediction)
```
![image](https://user-images.githubusercontent.com/114502827/192628718-a5effced-4410-4589-b584-531873e86bd9.png)

Шаг 2 На второй итерации отображаются значения параметров, значения потерь и эффекты визуализации после итерации
![image](https://user-images.githubusercontent.com/114502827/192630695-2e2d2805-3960-43ed-b4a9-6f99c38ffee7.png)

Шаг 3 Третья итерация показывает значения параметров, значения потерь и визуализацию после итерации
![image](https://user-images.githubusercontent.com/114502827/192630839-0de5793e-def8-4468-bfb8-90b40e20560d.png)

Шаг 4 На четвертой итерации отображаются значения параметров, значения потерь и эффекты визуализации
![image](https://user-images.githubusercontent.com/114502827/192630991-2790b022-c337-410b-8b7b-6d6fc21eecb9.png)

Шаг 5 Пятая итерация показывает значение параметра, значение потерь и эффект визуализации после итерации
![image](https://user-images.githubusercontent.com/114502827/192631119-9dbd740b-f749-4252-8a5f-7c61e2ca839c.png)

Шаг 6 10000-я итерация, показывающая значения параметров, потери и визуализацию после итерации
![image](https://user-images.githubusercontent.com/114502827/192631308-7a6c4358-0b2c-4e0d-a2dd-df6f0918eb7d.png)




## Задание 3
## Изучить код на Python и ответить на вопросы:
- Должна ли величина loss стремиться к нулю при изменении исходных данных? Ответьте на вопрос, 
приведите пример выполнения кода, который подтверждает ваш ответ.

 Да, величина loss будет стремится к нулю, например при изменении количества итераций.

   1 - ![image](https://user-images.githubusercontent.com/114502827/192634742-c2b2d48a-3bdc-4e41-ae54-f88ea22a3433.png)
 
   2 - ![image](https://user-images.githubusercontent.com/114502827/192634666-f2fb7e9a-846e-41c6-a6cb-9775753776ee.png)
 
10000 - ![image](https://user-images.githubusercontent.com/114502827/192634927-0157f726-395e-4daf-a426-d873b7eb1dbe.png)


- Какова роль параметра Lr? Ответьте на вопрос, приведите пример выполнения кода, который подтверждает ваш ответ. 
В качестве эксперимента можете изменить значение параметра.

Праметр Lr влияет на точность графика. Чем меньше параметр - тем выше точность 
![image](https://user-images.githubusercontent.com/114502827/192636929-5ac891be-4638-4abc-9e08-6e5ec82331a7.png)
![image](https://user-images.githubusercontent.com/114502827/192636980-4c717161-a4a1-4dbb-8a2e-f938d0881c95.png)




## Выводы

Абзац умных слов о том, что было сделано и что было узнано.
- были установлены Anaconda, VSCode и два раза Unity
- с трудом произошла настройка Unity и связки с VSCode
- была реализована программа "Hello World" на таких языках как Python и C#
- произошло знакомство с GitHub и создание своего первого репозитория

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**

