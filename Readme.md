
## 1. Запуск проекта

```sh
g++ -o main main.cpp
./main
```


## 2. Формирование тренировочного датасета

```sh
python3 generate.py > training_dataset.txt
```

- Идея generate.py:


```python
0: ["cats", "and", "dogs", "are", "mouse", "hiding", "friends", "from", "cat", "Mouse", "eats", "cheese", "next", "to"],

1: ["I", "football", "play", "have", "another", "with", "my", "Our", "is", "the", "team", "march", "friends", "called", "on"]
```

- Идея generate2.py - добавление большего кол-ва шума.

## 3. Тестовые данные

```

4 2
0
Cats and dogs are friends.
0
Mouse hiding from cat.
1
I play football with my friends.
1
Our football team is called the March cats.
Mouse eats cheese next to cats
I have friends on another football team. 

```
- Результат последних двух строк: 0 1 соответственно.

## 4. Замеры скорости работы


```sh
g++ -o benc benchmark.cpp
./benc < training_dataset.txt > training_result.txt
```


- 4 2
```
[BENCHMARK] Train time: 8.8065e-05 s
[BENCHMARK] Classify time (all test docs): 7.32553 s
[BENCHMARK] Preprocess call count: 6
[BENCHMARK] Total preprocessing time: 4.0926e-05 s
```

- 40 20
```
[BENCHMARK] Train time: 0.000530137 s
[BENCHMARK] Classify time (all test docs): 0.00058866 s
[BENCHMARK] Preprocess call count: 80
[BENCHMARK] Total preprocessing time: 0.000400932 s
```

- 400 200
```
[BENCHMARK] Train time: 0.00541744 s
[BENCHMARK] Classify time (all test docs): 0.00544363 s
[BENCHMARK] Preprocess call count: 800
[BENCHMARK] Total preprocessing time: 0.00413466 s
```

- 4000 2000

```
[BENCHMARK] Train time: 0.0325395 s
[BENCHMARK] Classify time (all test docs): 0.0405835 s
[BENCHMARK] Preprocess call count: 8000
[BENCHMARK] Total preprocessing time: 0.0288701 s
```
