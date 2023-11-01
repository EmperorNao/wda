
from os.path import join

from constants import LOCAL, RANDOM_STATE

from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer

from dataset import Dataset


if __name__ == "__main__":
    normalize = True
    dataset = Dataset()
    if normalize:
        dataset.from_csv(join(LOCAL, 'data/dataset.csv'))
    else:
        dataset.from_csv(join(LOCAL, 'data/simple_data.csv'))

    for k in [25, 50, 100, 150, 200, 250, 500]:
        pca = TruncatedSVD(k, random_state=RANDOM_STATE)
        vectorizer = TfidfVectorizer()

        pca.fit(vectorizer.fit_transform(dataset.data['title']))
        for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
            cls = SVC(class_weight='balanced', C=C, random_state=RANDOM_STATE)
            scores = cross_validate(cls,
                                    *shuffle(
                                        pca.transform(vectorizer.transform(dataset.train_data['title'])),
                                        dataset.train_data['target'],
                                        random_state=RANDOM_STATE
                                    ),
                                    cv=5,
                                    scoring='f1',
                                    return_train_score=True,
                                    )
            test_scores = scores['test_score']
            train_scores = scores['train_score']
            print(f"n_comps: {k}, "
                  f"C={C}, "
                  f"testF1 = {sum(test_scores) / len(train_scores)}, "
                  f"trainF1 = {sum(train_scores) / len(train_scores)}")
