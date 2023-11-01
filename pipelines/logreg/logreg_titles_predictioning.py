
from os.path import join


from constants import LOCAL, RANDOM_STATE

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from dataset import Dataset


if __name__ == "__main__":
    dataset = Dataset()
    dataset.from_csv(join(LOCAL, 'data/simple_data.csv'))

    C = 1
    vectorizer = TfidfVectorizer()
    cls = LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, C=C)

    # специально фитимся на train + test
    vectorizer.fit_transform(dataset.data['title'])
    cls.fit(vectorizer.transform(dataset.train_data['title']), dataset.train_data['target'])
    predictions = cls.predict(vectorizer.transform(dataset.test_data['title']))

    df = pd.DataFrame({'target': predictions, 'pair_id': dataset.test_data['pair_id']})
    df.to_csv(join(LOCAL, "data", f"logreg_titles_C{C}_predictions.csv"), header=True, index=False)