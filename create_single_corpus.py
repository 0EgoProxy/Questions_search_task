"""
Приводим тектс из корпуса текста"" к единому
Даный скрип производит чтение текстовых файлов из RusTextCorpus(сборник спаршенных новостей,
коментариев, диалогов) и объединяет их в один единый текстовый файл "file_data_root.txt".

В процессе обработки каждого файла производится разделение текста на отдельные предложения.
Предложения записываюстя в новый файл таким образом, что каждое отдельное предложение записывается построчно.
При этом последний символ предложения разделен сепаратором: "\t" и записывается в конце данной строчки.

Так же в процессе обработки корпуса считается:  количество предложений, количество вопросов.
Формируется словарь уникальных символов, на которые оканчиваются предложения, и производиться их подсчет.
"""

import json
from pathlib import Path
from typing import List

from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm

if __name__ == '__main__':
    dataset_statistics = {'sentence_count': 0,
                          'end_sent_count': {'?': 0, '.': 0, ';': 0, '!': 0, '!?': 0, '...': 0},
                          'end_sent': {'?': []}}

    name_of_direcoti_root_file = Path.cwd()
    home_path = Path.home()
    work_path = Path("/media/owl/Seagate Expansion Drive/RusTextCorpus/news/Fontanka/texts/example")

    global_sent_ind = 0
    doc_num = 0
    for doc in work_path.rglob("**/*.txt"):
        doc_num += 1
    # ------------------------------------------------------------------------------------------
    with open("file_data_root.txt", 'a') as root_file:
        for file_txt in tqdm(work_path.rglob("**/*.txt"), total=doc_num):

            with open(file_txt, 'r', encoding='utf-8') as text_in_file:
                for line in text_in_file:
                    if line != '':
                        sentences: List[str] = sent_tokenize(line)  # [предложение, предложение]
                        count_sent = len(sentences)
                        result_token_word = []
                        for ind, sent in enumerate(sentences):
                            result_token_word.append(word_tokenize(sent))  # [['слова', 'символы'],
                            # (если есть ещё предложения) ['слова', 'символы']]
                            sent_in_line: str = ' '.join(result_token_word[ind][:-1]) + '\t' + \
                                                result_token_word[ind][-1] + '\n'
                            root_file.write(sent_in_line)

                            global_sent_ind += 1
                            dataset_statistics['sentence_count'] += 1

                            if result_token_word[ind][-1] == ".":
                                dataset_statistics['end_sent_count']['.'] += 1
                            elif result_token_word[ind][-1] == "?":
                                dataset_statistics['end_sent_count']['?'] += 1
                                dataset_statistics['end_sent']['?'].append(global_sent_ind)
                            else:
                                if result_token_word[ind][-1] not in dataset_statistics['end_sent_count']:
                                    dataset_statistics['end_sent_count'][result_token_word[ind][-1]] = 1
                                    dataset_statistics['end_sent'][result_token_word[ind][-1]] = []
                                    dataset_statistics['end_sent'][result_token_word[ind][-1]].append(global_sent_ind)
                                else:
                                    dataset_statistics['end_sent_count'][result_token_word[ind][-1]] += 1

                text_in_file.close()
        root_file.close()

    with open('dataset_stats.json', 'w', encoding='utf-8') as statfile:
        json.dump(dataset_statistics, statfile)
