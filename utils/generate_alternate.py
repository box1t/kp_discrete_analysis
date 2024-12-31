import random

def generate_vocab(group_words, extra_size=50, common_size=10):
    """
    Генерирует «дополнительные» (extra) слова и «общие» (common) слова,
    которые могут попадаться в документах обоих классов.
    """
    all_used = set(group_words[0] + group_words[1])

    extra_words = set()
    while len(extra_words) < extra_size:
        word_length = random.randint(3, 8)
        word = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(word_length))
        if word not in all_used:
            extra_words.add(word)

    common_words = set()
    while len(common_words) < common_size:
        word_length = random.randint(3, 8)
        word = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(word_length))
        if word not in all_used:
            common_words.add(word)

    return group_words, list(extra_words), list(common_words)


def generate_data(
    group_words, 
    extra_words, 
    common_words,
    n_train_0, n_train_1, 
    n_test, 
    avg_words=10, 
    overlap_ratio=0.2
):
    """
    Генерирует тренировочные и тестовые данные.
    
    Формат возвращаемых данных:
      train_data: список кортежей (label, text)
      test_data:  список кортежей (label, text)
                  (label может использоваться для расчёта метрик, а text печатается)
    
    Параметры:
    -----------
    group_words : dict, например {0: [...], 1: [...]}
    extra_words : list, слова, не связанные напрямую с классами
    common_words: list, слова, встречающиеся в обоих классах
    n_train_0, n_train_1 : количество документов на тренинг для класса 0 и класса 1
    n_test : общее кол-во тестовых документов
    avg_words : средняя длина каждого документа (в словах)
    overlap_ratio : доля «заимствованных» из "другого" класса слов,
                    чтобы добавить шум и избежать 100% совпадения.
                    Например, если overlap_ratio=0.2, то ~20% слов для класса 0 
                    берётся из набора класса 1 и наоборот.
    """
    train_data = []
    test_data = []

    def make_doc(label):
        """
        label = 0 или 1
        Генерируем документ так, чтобы часть слов была из group_words[label],
        часть (overlap_ratio) – из group_words[1-label], 
        пара слов – из common_words, и немного – из extra_words.
        """
        k = avg_words
        # сколько слов возьмём из своей группы
        main_count = int(k * (1.0 - overlap_ratio))
        # сколько слов возьмём из противоположной группы
        overlap_count = int(k * overlap_ratio * 0.5)  # половину overlap возьмём, остальное можно докинуть в common/extra
        
        # остаток пойдёт на общие/экстра-слова
        leftover_count = k - main_count - overlap_count
        
        # Выбираем из своей группы
        gw = random.sample(group_words[label], min(main_count, len(group_words[label])))

        # Выбираем из «чужой» группы
        gw_other = random.sample(group_words[1 - label], min(overlap_count, len(group_words[1 - label])))

        # Выбираем из common_words/extra_words
        common_part = random.sample(common_words, min(leftover_count, len(common_words)))
        # если осталось место, добавим пару случайных из extra
        leftover_count2 = leftover_count - len(common_part)
        extra_part = []
        if leftover_count2 > 0:
            extra_part = random.sample(extra_words, min(leftover_count2, len(extra_words)))

        words = gw + gw_other + common_part + extra_part
        random.shuffle(words)
        return ' '.join(words)

    # Генерация обучающих данных (класс 0)
    for _ in range(n_train_0):
        text = make_doc(0)
        train_data.append((0, text))

    # Генерация обучающих данных (класс 1)
    for _ in range(n_train_1):
        text = make_doc(1)
        train_data.append((1, text))

    # Генерация тестовых данных
    # Для метрик нам нужна истинная метка, так что вернём (label, text).
    for _ in range(n_test):
        label = random.choice([0, 1])
        text = make_doc(label)
        test_data.append((label, text))

    return train_data, test_data


if __name__ == "__main__":
    # --------------------------------
    # 1. Определяем базовый набор слов
    # --------------------------------
    # Ваша идея: класс 0 — "про котов и мышей", класс 1 — "про футбол".
    group_words = {
        0: ["cats", "dogs", "mouse", "hiding", "cheese", "friends", "next", "to", "cat", "are", "from", "eats"],
        1: ["football", "play", "team", "march", "called", "on", "game", "with", "my", "friends", "our", "is"]
    }

    # --------------------------------
    # 2. Генерация словарей
    # --------------------------------
    # extra_size, common_size — сколько слов генерировать "случайных"
    group_words, extra_words, common_words = generate_vocab(
        group_words,
        extra_size=50,
        common_size=10
    )

    # --------------------------------
    # 3. Задаём параметры эксперимента
    # --------------------------------
    N_train_0 = 20  # Количество обучающих текстов класса 0
    N_train_1 = 20  # Количество обучающих текстов класса 1
    N_test    = 20  # Количество тестовых текстов

    avg_words     = 10    # Примерная длина одного текста
    overlap_ratio = 0.2   # Доля "заимствованных" слов из чужого класса

    # --------------------------------
    # 4. Генерация данных
    # --------------------------------
    train_data, test_data = generate_data(
        group_words,
        extra_words,
        common_words,
        N_train_0, 
        N_train_1, 
        N_test,
        avg_words=avg_words,
        overlap_ratio=overlap_ratio
    )

    # --------------------------------
    # 5. Вывод результатов в нужном формате
    # --------------------------------
    #  Сначала: (N_train_0 + N_train_1)  N_test
    #  Затем — пары (label, text) для train
    #  И наконец — тексты для test (без меток).
    #
    #  При этом для себя можно сохранить test_labels отдельно.
    print(f"{N_train_0 + N_train_1} {N_test}")

    # Сначала обучающие (метка + текст)
    for (label, text) in train_data:
        print(label)
        print(text)

    # Затем тестовые (только текст!)
    # Но для проверки мы сохраним метки в отдельный список `test_labels`.
    test_labels = []
    for (label, text) in test_data:
        test_labels.append(label)
        print(text)

    # Если нужно проверять метрики на тесте, используйте test_labels
    # вместе с предсказанными метками вашей C++ программы.
    # Например, вы можете записать test_labels в файл, чтобы потом
    # сравнить их с выводом.
