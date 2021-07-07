import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def class_return(sign):
    return 1 if sign == '?' else 0


if __name__ == '__main__':
    # TODO: rewrite and optimize
    print("Загружаем датасет ...")
    path_root = Path('/')
    work_dataset_for_forest = pd.read_csv(path_root / 'dataset_of_questions' / 'work_dataset.csv', sep='\t')
    work_dataset_for_forest = work_dataset_for_forest.drop('Unnamed: 0', axis=1)
    work_dataset_for_forest.rename(columns={'Line informatation': 'line_information'}, inplace=True)
    work_dataset_for_forest['label'] = work_dataset_for_forest['Sign'].apply(class_return)
    print()
    print("Датасет успешно загружен!", '\n')

    print("TfidfVectorizer обучается ...", '\n')
    fit_object_forest = TfidfVectorizer()
    fit_object_forest.fit(work_dataset_for_forest['line_information'])
    with open('fit_vectorizer', 'wb') as fit_file:
        pickle.dump(fit_object_forest, fit_file)
    print("Веткторайзер обучен и записан как  ** fit_vectorizer **")

    train_data, test_data, train_label, test_label = train_test_split(work_dataset_for_forest,
                                                                      work_dataset_for_forest['label'],
                                                                      test_size=0.3,
                                                                      random_state=0)

    fit_data_forest = fit_object_forest.transform(train_data['line_information'])
    param_test1 = {'max_depth': [4, 5, 6], 'min_child_weight': [4, 5, 6]}

    gscv_object1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                                        n_estimators=500,
                                                        max_depth=7,
                                                        gamma=0,
                                                        subsample=0.8,
                                                        colsample_bytree=0.8,
                                                        objective='binary:logistic',
                                                        nthread=4,
                                                        scale_pos_weight=1,
                                                        seed=27),
                                param_grid=param_test1, scoring='f1_macro', n_jobs=6, cv=3)

    print("Начинаем обучать случайный лес, с заданными параметрами. \n  это может занять некоторое время ...")
    gscv_object1.fit(fit_data_forest, train_label)

    print(f'Лучшие параметры модели: {gscv_object1.best_params_}')
    print(f"Лучшая оценка: {gscv_object1.best_score_}")

    test_transfrotm = fit_object_forest.transform(test_data['line_information'])
    best_model = gscv_object1.best_estimator_
    predict_answer_f = best_model.predict(test_transfrotm)
    rc_score_test = roc_auc_score(test_label, predict_answer_f)
    print(rc_score_test)

    disp = plot_precision_recall_curve(gscv_object1, test_transfrotm, test_label)
    disp.ax_.set_title(f'2-class Precision-Recall curve: AP={0.88}')

    classification_assessment = classification_report(test_label, predict_answer_f, output_dict=True)

    print("Выводим основные метрики модели: \n")
    print(f"Словарь метрик: {classification_assessment}")
