import json
from pathlib import Path
from typing import List, Dict, Tuple, TextIO

from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm


def check_file(root: TextIO, file: str, stat_dict: Dict, hard_stat: Dict, counter: int) -> Tuple[Dict, Dict, int]:
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

                    if result_token_word[ind][-1] in stat_dict['end_sent_count']:
                        stat_dict['end_sent_count'][result_token_word[ind][-1]] += 1
                        hard_stat['ind_sent'][result_token_word[ind][-1]].append(counter)

        text_in_file.close()

    return stat_dict, hard_stat, counter


if __name__ == '__main__':
    # make dictionaries
    dataset_statistics = {'sentence_count': 0,
                          'end_sent_count': {'?': 0, '.': 0, ';': 0, '!': 0, ',': 0, '...': 0, ':': 0}}
    ind_hard_stats: Dict = {'ind_sent': {'?': [], '.': [], ';': [], '!': [], ',': [], '...': [], ':': []}}

    # parse all necessary file paths
    root_path = Path.cwd().joinpath('dataset_of_questions')
    root_path.mkdir(parents=False, exist_ok=False)

    # todo: write comment
    data_root = Path("/media/owl/Seagate Expansion Drive/RusTextCorpus")
    all_news_path = data_root.joinpath('news')
    all_social_path = data_root.joinpath('social')
    all_nplus_path = data_root.joinpath('NPlus1')

    # todo: write comment
    all_paths = [doc for doc in all_news_path.rglob("*/texts/*.txt")]
    all_paths.extend([doc for doc in all_social_path.rglob("**/texts/*.txt")])
    all_paths.extend([doc for doc in all_nplus_path.rglob("**/texts/*.txt")])
    doc_num = len(all_paths)

    # start parsing all files
    root_file = root_path.joinpath('root_dataset.txt').open('a', encoding='utf-8')
    global_sent_ind = 0
    flow_open = True
    for i, file_txt in tqdm(enumerate(all_paths), total=doc_num):
        if i % 10000 == 0:
            dataset_statistics, ind_hard_stats, global_sent_ind = check_file(root=root_file,
                                                                             file=file_txt,
                                                                             stat_dict=dataset_statistics,
                                                                             hard_stat=ind_hard_stats,
                                                                             counter=global_sent_ind)
            root_file.close()
            flow_open = False
        else:
            if flow_open:  # if flow_open is True:
                dataset_statistics, ind_hard_stats, global_sent_ind = check_file(root=root_file,
                                                                                 file=file_txt,
                                                                                 stat_dict=dataset_statistics,
                                                                                 hard_stat=ind_hard_stats,
                                                                                 counter=global_sent_ind)
            else:
                root_file = open("root_dataset.txt", 'a', encoding='utf-8')
                flow_open = True
                dataset_statistics, ind_hard_stats, global_sent_ind = check_file(root=root_file,
                                                                                 file=file_txt,
                                                                                 stat_dict=dataset_statistics,
                                                                                 hard_stat=ind_hard_stats,
                                                                                 counter=global_sent_ind)

    with open('dataset_stats.json', 'w', encoding='utf-8') as statfile:
        json.dump(dataset_statistics, statfile)

    with open('ind_hard_stats', 'w', encoding='utf-8') as hard_stat_file:
        json.dump(ind_hard_stats, hard_stat_file)
