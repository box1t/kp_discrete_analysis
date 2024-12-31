import random

def generate_vocab(group_words, extra_size=100, common_size=10):
    """
    Генерирует полный словарь, включая дополнительные и общие слова.
    """
    extra_words = set()
    while len(extra_words) < extra_size:
        word_length = random.randint(3, 8)
        extra_word = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(word_length))
        if extra_word not in group_words[0] and extra_word not in group_words[1]:
            extra_words.add(extra_word)

    common_words = set()
    while len(common_words) < common_size:
        word_length = random.randint(3, 8)
        common_word = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(word_length))
        if common_word not in group_words[0] and common_word not in group_words[1]:
            common_words.add(common_word)

    return group_words, list(extra_words), list(common_words)

def generate_data(group_words, extra_words, common_words, n_train_0, n_train_1, n_test, avg_words=10, overlap_ratio=0.2):
    """
    Генерирует тренировочные и тестовые данные в формате:
    1. Сначала все тексты класса 0.
    2. Затем все тексты класса 1.
    """
    train_data = []
    test_data = []

    # Генерация тренировочных данных для класса 0
    for _ in range(n_train_0):
        words = random.sample(group_words[0], avg_words - 2) + random.sample(common_words, 2)
        random.shuffle(words)
        train_data.append((0, ' '.join(words)))

    # Генерация тренировочных данных для класса 1
    for _ in range(n_train_1):
        words = random.sample(group_words[1], avg_words - 2) + random.sample(common_words, 2)
        random.shuffle(words)
        train_data.append((1, ' '.join(words)))

    # Генерация тестовых данных
    for _ in range(n_test):
        group_choice = random.choice([0, 1])
        group_words_sample = random.sample(group_words[group_choice], avg_words - 3)
        common_sample = random.sample(common_words, 2)
        extra_sample = random.sample(extra_words, 1)
        words = group_words_sample + common_sample + extra_sample
        random.shuffle(words)
        test_data.append(' '.join(words))

    return train_data, test_data

if __name__ == "__main__":
    # Определяем группы слов для каждого класса
    group_words = {
        0: ["cats", "dogs", "mouse", "hiding", "cheese", "friends", "next", "to", "cat", "are", "from", "eats"],
        1: ["football", "play", "team", "march", "called", "on", "game", "with", "my", "friends", "our", "is"]
    }

    # Генерация словарей
    group_words, extra_words, common_words = generate_vocab(group_words, extra_size=50, common_size=10)

    # Параметры генерации
    N_train_0 = 2  # Количество тренировочных текстов для класса 0
    N_train_1 = 2  # Количество тренировочных текстов для класса 1
    N_test = 2     # Количество тестовых текстов

    # Генерация данных
    train_data, test_data = generate_data(group_words, extra_words, common_words, N_train_0, N_train_1, N_test)

    # Вывод результатов
    print(f"{N_train_0 + N_train_1} {N_test}")
    for label, doc in train_data:
        print(label)
        print(doc)

    for doc in test_data:
        print(doc)
