import numpy as np
import pandas as pd
import os

def merge_datasets_to_single_file(output_train_file='merged_train.csv', output_test_file='merged_test.csv'):
    datasets = ['a', 'b', 'd', 'e', 'k', 'n', 'o', 'u', 'y', 'z']
    merged_train_data = []
    merged_test_data = []

    for idx, dataset in enumerate(datasets):
        # Путь к тренировочным данным
        train_data_path = os.path.join('class_datasets', dataset, f'{dataset}_train.csv')
        # Путь к тестовым данным
        test_data_path = os.path.join('class_datasets', dataset, f'{dataset}_test.csv')

        # Загружаем тренировочные данные
        train_df = pd.read_csv(train_data_path)
        train_df['class'] = idx  # Добавляем столбец с меткой класса для тренировочных данных
        merged_train_data.append(train_df)

        # Загружаем тестовые данные
        test_df = pd.read_csv(test_data_path)
        test_df['class'] = idx  # Добавляем столбец с меткой класса для тестовых данных
        merged_test_data.append(test_df)

    # Объединяем все тренировочные данные в один DataFrame
    merged_train_df = pd.concat(merged_train_data, ignore_index=True)
    
    # Перемешиваем тренировочные данные
    merged_train_df = merged_train_df.sample(frac=1).reset_index(drop=True)
    
    # Объединяем все тестовые данные в один DataFrame
    merged_test_df = pd.concat(merged_test_data, ignore_index=True)
    
    # Перемешиваем тестовые данные
    merged_test_df = merged_test_df.sample(frac=1).reset_index(drop=True)

    # Сохраняем тренировочные данные в файл
    merged_train_df.to_csv(output_train_file, index=False)
    print(f"Training data merged and saved to {output_train_file}")

    # Сохраняем тестовые данные в файл
    merged_test_df.to_csv(output_test_file, index=False)
    print(f"Test data merged and saved to {output_test_file}")

# Запуск объединения тренировочных и тестовых файлов
merge_datasets_to_single_file()
