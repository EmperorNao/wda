
from os.path import join

from constants import LOCAL, RANDOM_STATE

import pandas as pd

from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from dataset import SimpleDataset


if __name__ == "__main__":
    dataset = SimpleDataset()
    dataset.from_csv(join(LOCAL, 'data/simple_data.csv'))

    k = 100
    C = 0.1
    pca = TruncatedSVD(n_components=k, random_state=RANDOM_STATE)
    vectorizer = TfidfVectorizer()
    cls = SVC(class_weight='balanced', C=C, random_state=RANDOM_STATE)

    # специально фитимся на train + test
    pca.fit(vectorizer.fit_transform(dataset.data['title']))
    cls.fit(pca.transform(vectorizer.transform(dataset.train_data['title'])), dataset.train_data['target'])
    predictions = cls.predict(pca.transform(vectorizer.transform(dataset.test_data['title'])))

    df = pd.DataFrame({'target': predictions, 'pair_id': dataset.test_data['pair_id']})
    df.to_csv(join(LOCAL, "data", f"SVM_titles_k{k}_C{C}_predictions.csv"), header=True, index=False)
