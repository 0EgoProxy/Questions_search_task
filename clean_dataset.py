import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def chek_rubbish(dataframe, chek_list):
    count_trash = 0
    for sign_ in dataframe['Sign']:
        if sign_ not in chek_list:
            count_trash += 1

    return count_trash


if __name__ == '__main__':
    # TODO: rewrite and optimize
    # /'root_dataset.txt' - имя файла
    print('Считываем файл и переводим его в датафрейм ... ')
    path_work_file = Path('/dataset/root_dataset.csv')
    data_sentenes = pd.read_csv(path_work_file, sep='\t', names=['line_information', 'Sign'])

    """
    # Удаление всех строк, если содержание столбца "Sign" нет в списке допустимых символов |signs: List |
    """
    signs = ['?', '.', ';', '!', ',', '...', ':']
    data_sentenes = data_sentenes.loc[data_sentenes['Sign'].isin(signs)]

    # --------------------------------------------------------------------------------------------
    """ Делаем проверку на остаточный мусор."""
    print('Проверяем на мусор ...')
    nums_rubbish = chek_rubbish(data_sentenes, signs)
    print(f'Предложений с мусором: {nums_rubbish}')

    # --------------------------------------------------------------------------------------------
    """ Удаление пропущенных значений. """

    print(" Проверяем, есть ли пропущенные значения, если есть - удаляем.")
    print(f"Пропущенных значений: {data_sentenes['line_information'].isnull().sum()}")
    data_sentenes = data_sentenes.dropna()
    print(f"Пропущенных значений после удаления: {data_sentenes['line_information'].isnull().sum()}")
    # --------------------------------------------------------------------------------------------
    """ Предобработка. """

    print("Выводим количество каждого обьекта после сортировки.")
    sign_dict = {'end_sent_count': {'?': 0, '.': 0, ';': 0, '!': 0, ',': 0, '...': 0, ':': 0}}

    count_error = 0
    for sign in data_sentenes['Sign']:
        if sign in signs:
            sign_dict['end_sent_count'][sign] += 1

    print(sign_dict)
    print(f'Предложение с мусором: {chek_rubbish(data_sentenes, signs)}')

    """
       **Балансировка классов**
    Так как у нас задача бинарной классификации, а именно: важно понять предложение вопросительный характер - "1" 
    или нет - "0"(То есть все прочие знаки). То Появляется необходимость в соблюдении баланса классво. 
    Если говорить конкретнее: "?": 541 126 обьектов КЛАССА - "1". Столько же обьектов сделаем и для КЛАССА - "0". 
    Для упрощения сделаем 540 000 обьектов для классов "1", "0"
    
    При этом важно сохранить пропорцию объектов из класса "0". Так как в реальных примерах для модели вероятность 
    встретить предложение определенного характера будет совпадать с нашими данным - исходя их статистики.
    
    Для этого посчитаем процентное соотношение обьектов различных знаков по отношению ко всем предложениям 
    за вычетом вопросительных, их в отдельную выборку. после чего через train.test.split поставим необходимое 
    соотношение в оставшейся выборке и соединим её с уже имеющейся выборкой вопросительных знаков.
    """

    """
    ----------------------------------------------------------------------------------------------------------
    Варианты удаления строки из dataframe: 
    Вариант 1: фильтруем по одному значению: df = df.loc[df['STP'] != 1005092] 
    Вариант 2: можно указать несколько значений: df = df.loc[~df['STP'].isin([1005092])] 
    Вариант 3: фильтруем по одному значению: df = df.query("STP != 1005092") 
    Вариант 4: можно указать несколько значений: 
        df = df.query("STP not in [1005092, ...]") Вариант 5: df = df.drop(np.where(df['STP'] == 1005092)[0])
    ----------------------------------------------------------------------------------------------------------
    """

    data_of_question = data_sentenes.loc[data_sentenes['Sign'] == '?']  # Создаем датасет содержащий только вопросы

    # Избавляемся в основном датасете предложений от предложений с вопросом
    data_sentenes = data_sentenes.loc[data_sentenes['Sign'] != '?']
    data_of_question.index = np.arange(len(data_of_question))  # даем новые, корректные индексы

    print(f"Размер выбокри без вопросов: {data_sentenes['Sign'].size}")
    print(f"Размер выборки с вопросами: {data_of_question['Sign'].size}")

    # Проверяем, остались ли вопросы в основном датасете "data_sentenes"
    print('Датасет не отсортирован') if "?" in data_sentenes['Sign'] else print('Датасет читс')
    # --------------------------------------------------------------------------------------------

    """
    Непосредственно создаем единый датасет содержащий одинаковое количество объектов классов
    с пропорциональной действительность.
    """

    percent_qestion_of_sent = round(data_of_question['Sign'].size / data_sentenes['Sign'].size, 4)

    print(
        f"Процент вопросительных предложений по отношению к числу предложений без них {percent_qestion_of_sent * 100}%")

    train_sent_data, test_sent_data = train_test_split(data_sentenes,
                                                       test_size=percent_qestion_of_sent,
                                                       stratify=data_sentenes.Sign)

    work_dataset = pd.concat([test_sent_data, data_of_question], axis=0)
    work_dataset = shuffle(work_dataset)  # Выполняем перемешивание
    work_dataset.index = np.arange(len(work_dataset))  # Делаем коректные индексы.

    percent_series_sign = work_dataset['Sign'].value_counts(normalize=True).to_dict()
    print(f"Соотношение знаков: {percent_series_sign}")
    # --------------------------------------------------------------------------------------------
    work_dataset_statistic = {'shape': work_dataset.shape,
                              'value_counts': percent_series_sign,
                              'name_culumns': list(work_dataset.columns)}

    """ Переводим датафрейм в формат ** csv ** """
    work_dataset.to_csv('/home/owl/PycharmProjects/Questions_search_task/dataset_of_questions/work_dataset.csv',
                        sep='\t')

    with open('dataset/work_dataset_statistic.json', 'w', encoding='utf-8') as statfile:
        json.dump(work_dataset_statistic, statfile)
