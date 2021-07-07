import pickle
from pathlib import Path
from typing import Dict, List, Union


class LogRegPipeline:
    def __init__(self, vectorizer_path: Union[str, Path], model_path: Union[str, Path]):
        # загружаем в корку класса обученный на рабочем датасете векторайзер.
        with open(vectorizer_path, 'rb') as file:
            self.work_vectorizer = pickle.load(file)

        with open(model_path, 'rb') as file:
            self.best_model = pickle.load(file)

        self.out_names = {0: 'NQ', 1: '?'}

    def predict_question(self, text: Union[str, List[str]], print_report: bool = False) -> Union[List[str],
                                                                                                 Dict[str, str]]:
        if isinstance(text, str):
            text = [text]

        vec_transform = self.work_vectorizer.transform(text)
        predict_label = self.best_model.predict(vec_transform)

        if print_report:
            return self.dict_report(text, [self.out_names[x] for x in predict_label])
        else:
            return [self.out_names[x] for x in predict_label]

    @staticmethod
    def dict_report(text: List[str], labels: List):
        return {key: value for key, value in zip(text, labels)}


if __name__ == '__main__':
    root_path = Path(__file__).resolve().parent
    path2vectorizer = root_path / 'checkpoints' / 'fit_vectorizer.pkl'
    path2model = root_path / 'checkpoints' / 'best_model_logreg.pkl'

    pipeline = LogRegPipeline(path2vectorizer, path2model)

    # start tests
    print()
    print('test on list sentence')
    list_sentences = ['Так что, завтра мы идем куда-нибудь',
                      'ИНН: 0000000002839/ ФИО: Григорьев Григорий Григорьевич',
                      'Как думаешь завтра дождь пойдет',
                      'гулять будемм нет))']

    print(pipeline.predict_question(list_sentences))
    print(pipeline.predict_question(list_sentences, print_report=True))

    # test single sentence
    print()
    print('test single sentence')
    print(pipeline.predict_question('ехал грека через реку'))
    print(pipeline.predict_question('видит грека в реке рак', print_report=True))
