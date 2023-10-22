from os.path import join
import pandas as pd
import numpy as np
from dataset import SimpleDataset
from preprocessing import count_group_intersection_features

LOCAL = "~/personal/wda"

train_dataset = SimpleDataset(
    join(LOCAL, "data/docs_titles.tsv"),
    join(LOCAL, "data/train_groups.csv"),
    train=True
)

test_dataset = SimpleDataset(
    join(LOCAL, "data/docs_titles.tsv"),
    join(LOCAL, "data/test_groups.csv"),
    train=False
)


train_data = count_group_intersection_features(train_dataset)
test_data = count_group_intersection_features(test_dataset)

#train_data[['pair_id', 'count', 'group_avg_count', 'target']]

features = ['count', 'group_avg_count']
target = ['target']

X_train = train_data[features].to_numpy()
y_train = train_data[target].to_numpy().ravel()

X_test = test_data[features].to_numpy()
pairs_test = test_data['pair_id'].to_numpy()

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report

clf = svm.SVC()
clf.fit(X_train, y_train)

predictions_train = clf.predict(X_train)
print(np.unique(y_train))
print(np.unique(predictions_train))
print(classification_report(y_train, predictions_train))

predictions = clf.predict(X_test)

df = pd.DataFrame({'pair_id': pairs_test, 'target': predictions})
df.to_csv("handmade_features.csv", index=None)
