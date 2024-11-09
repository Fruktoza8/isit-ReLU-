import numpy as np
import pandas as pd
from collections import Counter

def load_data(train_file='merged_train.csv', test_file='merged_test.csv'):
    # Загрузка тренировочных данных
    train_df = pd.read_csv(train_file)
    
    # Преобразование признаков в числовой формат (float) и обработка ошибок
    # Убедимся, что Array колонка обрабатывается как строка и преобразуется в массив чисел
    X_train = np.array([np.fromstring(arr, sep=',') for arr in train_df['Array']])
    y_train = train_df['class'].values
    
    # Загрузка тестовых данных
    test_df = pd.read_csv(test_file)
    X_test = np.array([np.fromstring(arr, sep=',') for arr in test_df['Array']])
    y_test = test_df['class'].values

    # Нормализация данных
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Перемешивание данных
    train_indices = np.random.permutation(len(X_train))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    test_indices = np.random.permutation(len(X_test))
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    # Проверка распределения классов
    print("Train class distribution:", Counter(y_train))
    print("Test class distribution:", Counter(y_test))

    print(f"Train Data shape: {X_train.shape}, Train Labels shape: {y_train.shape}")
    print(f"Test Data shape: {X_test.shape}, Test Labels shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

# Пример использования:
# X_train, y_train, X_test, y_test = load_data('merged_train.csv', 'merged_test.csv')
