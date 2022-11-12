# DA-in-GameDev-lab3-3
# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #3 выполнил(а):
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
познакомиться с программными средствами для создания системы машинного обучения и ее интеграции в Unity.

## Задание 1
### Реализовать систему машинного обучения в связке Python - Google-Sheets – Unity. 
Ход работы:
- Создаем новый пустой 3D проект на Unity.
- Скачаваем папку с ML агентом. Вы найдете ее в облаке с исходными файлами к лабораторной работе – ml-agents-release_19.
- В созданный проект добавляем ML Agent, выбрав Window - Package Manager - Add Package from disk. Последовательно добавляем .json – файлы:                                       ml-agents-release_19 / com,unity.ml-agents / package.json
	ml-agents-release_19 / com,unity.ml-agents.extensions / package.json
	
![image](https://user-images.githubusercontent.com/114502827/201472588-0ea81d10-41d0-4a93-b6df-63ff8793a016.png)

![image](https://user-images.githubusercontent.com/114502827/201473184-a74b8e76-5012-443c-a0d8-67c942c76baf.png)

- Далее запускаем Anaconda Prompt для возможности запуска команд через консоль и пишем серию команд для создания и активации нового ML-агента, а также для скачивания необходимых библиотек

- Создаем на сцене плоскость, куб и сферу так, как показано на рисунке ниже. Создайте простой C# скрипт-файл и подключите его к сфере
 ![image](https://user-images.githubusercontent.com/114502827/201473252-ad468dea-e58d-4595-8d7f-b3b75f47c705.png)

- Добавляем в скрипт код:
 ```C#
 using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if(distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}

```
Запускае  MLAgent
![image](https://user-images.githubusercontent.com/114502827/201473898-9967d4a7-665f-49c5-a472-9286ae13eea4.png)

- Сделаем 3, 9, 27 копий модели «Плоскость-Сфера-Куб», запустим симуляцию сцены и понаблюдаем за результатом обучения модели.
![image](https://user-images.githubusercontent.com/114502827/201475505-de1dec25-6bc0-4553-982b-1bce11ae2867.png)

![image](https://user-images.githubusercontent.com/114502827/201475700-6c9edfc7-f656-4efe-96d9-8a29435a0f86.png)

![image](https://user-images.githubusercontent.com/114502827/201476366-4d9fde5f-4f47-47f8-a039-7074edadc13b.png)


##Задание 2

#Подробно опишите каждую строку файла конфигурации нейронной сети, доступного в папке с файлами проекта. Самостоятельно найдите информацию о компонентах Decision Requester, Behavior Parameters, добавленных на сфере.

```C#
behaviors: 
  RollerBall: 
    trainer_type: ppo #Тип обучения
    hyperparameters: 
      batch_size: 10 #объем данных (количество строк), загружаемый в память за один раз
      buffer_size: 100 #Количество опытов, которые необходимо провести перед обновлением модели
      learning_rate: 3.0e-4 #Коэффициент скорости обучения
      beta: 5.0e-4 #Сила регуляризации энтропии
      epsilon: 0.2 #Вещественное число, добавляемое к дисперсии, чтобы избежать деления на ноль
      lambd: 0.99 #Определяет насколько агент полагается на свою текущую оценку стоимости при вычислении обновленной оценки.
      num_epoch: 3 #Число эпох
      learning_rate_schedule: linear #То как будет меняться скорость обучения
    network_settings: 
      normalize: false #Определяет использование нормализации
      hidden_units: 128 #Колличество нейронов в одном слое сети
      num_layers: 2 #Колличество слоев нейронной сети
    reward_signals:
      extrinsic:
        gamma: 0.99 #Коэффициент дисконтирования для будущих вознаграждений
        strength: 1.0 #Коэффициент, на который умножается необработанное вознаграждение
    max_steps: 500000 #Максимальное число итераций
    time_horizon: 64 #Колличество шагов опыта, которые нужно собрать для агента, перед добавлением в буфер 
    summary_freq: 10000 #Колличество итераций, после которых будет промежуточный результат
    
```

Decision Requester - нужен для запрашивания решения через регулярные промежутки времени
Behavior Parameters - определяет принятие решений объектом


##Задание 3
#Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и в первом задании, случайно изменять координаты на плоскости. 

  ```C#
  using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }


    public Transform Target;
    public Transform TargetT;
    public float del = 1.42f;


    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
        TargetT.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        var middle = GetMiddle(Target.localPosition, TargetT.localPosition)
        sensor.AddObservation(middle);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);
        var middle = GetMiddle(Target.localPosition, TargetT.localPosition);

        float distanceToMid = Vector3.Distance(localPosition, middle);

        if(distanceToMid < del)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}

```
![image](https://user-images.githubusercontent.com/114502827/201481994-b963066e-3bf1-4efd-b886-3a4227afc265.png)


## Выводы

#то такое игровой баланс.
В однопользовательских играх:
Баланс в однопользовательских играх определяет сложность игры, сложность взаимоотношений объектов между собой во времени. Каждый уровень в игре должен быть сложнее предыдущего, тогда игрок будет ощущать фан, рост своего скилла, изменения и вызовы со стороны игры.

В многопользовательских играх:
Определение преимущества той или иной стороны — когда игроки получают разное оружие, выбирают персонажей различных классов. Игра является сбалансированной, когда классы персонажей, которые выбирают игроки, сбалансированы друг с другом. То есть, к примеру, нет никаких имбовых юнитов, когда можно получить большее преимущество при небольшом скилле.

#Как системы машинного обучения могут быть использованы для того, чтобы его скорректировать.
Главным преимущестовом нейронных сетей является их гибкость и адаптивность. Благодяря этому нейросети vогут управлять некоторым аспектом игрового баланса, например экономикой, и дать ей возможность обучаться, тем самым облегчить работу разработчика. Машинное обучение может мгновенно отреагировать на некоторый триггер в игровом процессе и сразу же донести информацию до разработчика.


## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
