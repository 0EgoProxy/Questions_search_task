import json
from pathlib import Path
from typing import List, Dict, Tuple, TextIO

from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm


def check_file(root: TextIO, file: str, stat_dict: Dict, counter: int) -> Tuple[Dict, int]:
    with open(file, 'r', encoding='utf-8') as text_in_file:
        for line in text_in_file:
            if line != '':
                sentences: List[str] = sent_tokenize(line)  # [предложение, предложение]
                result_token_word = []
                for ind, sent in enumerate(sentences):
                    result_token_word.append(word_tokenize(sent))  # [['слова', 'символы'],
                    # (если есть ещё предложения) ['слова', 'символы']]
                    sent_in_line = ' '.join(result_token_word[ind][:-1]) + '\t' + result_token_word[ind][-1] + '\n'
                    root.write(sent_in_line)

                    counter += 1
                    stat_dict['sentence_count'] += 1

                    if result_token_word[ind][-1] == ".":
                        stat_dict['end_sent_count']['.'] += 1
                    elif result_token_word[ind][-1] == "?":
                        stat_dict['end_sent_count']['?'] += 1
                        stat_dict['end_sent']['?'].append(counter)
                    else:
                        if result_token_word[ind][-1] not in stat_dict['end_sent_count']:
                            stat_dict['end_sent_count'][result_token_word[ind][-1]] = 1
                            stat_dict['end_sent'][result_token_word[ind][-1]] = []
                            stat_dict['end_sent'][result_token_word[ind][-1]].append(counter)
                        else:
                            stat_dict['end_sent_count'][result_token_word[ind][-1]] += 1
        text_in_file.close()

    return stat_dict, counter


if __name__ == '__main__':
    dataset_statistics = {'sentence_count': 0,
                          'end_sent_count': {'?': 0, '.': 0, ';': 0, '!': 0, '!?': 0, '...': 0},
                          'end_sent': {'?': []}}

    name_of_direcoti_root_file = Path.cwd()
    home_path = Path.home()
    work_path = Path("/media/owl/Seagate Expansion Drive/RusTextCorpus/news/Fontanka/texts")

    global_sent_ind = 0
    doc_num = 0
    for doc in work_path.rglob("**/*.txt"):
        doc_num += 1

    root_file = open("root_dataset.txt", 'a', encoding='utf-8')
    flow_open = True
    for i, file_txt in tqdm(enumerate(work_path.rglob("**/*.txt")), total=doc_num):
        if i % 10000 == 0:
            dataset_statistics, global_sent_ind = check_file(root=root_file, file=file_txt,
                                                             stat_dict=dataset_statistics, counter=global_sent_ind)
            root_file.close()
            flow_open = False
        else:
            if flow_open:  # if flow_open is True:
                dataset_statistics, global_sent_ind = check_file(root=root_file, file=file_txt,
                                                                 stat_dict=dataset_statistics, counter=global_sent_ind)
            else:
                root_file = open("root_dataset.txt", 'a', encoding='utf-8')
                flow_open = True
                dataset_statistics, global_sent_ind = check_file(root=root_file, file=file_txt,
                                                                 stat_dict=dataset_statistics, counter=global_sent_ind)

    with open('dataset_stats.json', 'w', encoding='utf-8') as statfile:
        json.dump(dataset_statistics, statfile)
