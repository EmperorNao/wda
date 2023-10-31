
from os.path import join

from constants import LOCAL, RANDOM_STATE

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer

from dataset import Dataset


if __name__ == "__main__":
    dataset = Dataset()
    dataset.from_csv(join(LOCAL, 'data/simple_data.csv'))
    vectorizer = TfidfVectorizer()
    vectorizer.fit(dataset.train_data['title'])

    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        cls = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, C=C, max_iter=10000)
        scores = cross_validate(cls,
                                *shuffle(
                                    vectorizer.transform(dataset.train_data['title']),
                                    dataset.train_data['target'],
                                    random_state=RANDOM_STATE
                                ),
                                cv=5,
                                scoring='f1',
                                return_train_score=True,
                                )
        test_scores = scores['test_score']
        train_scores = scores['train_score']
        print(f"C = {C}, "
              f"testF1 = {sum(test_scores) / len(train_scores)}, "
              f"trainF1 = {sum(train_scores) / len(train_scores)}")
