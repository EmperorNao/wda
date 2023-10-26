import pandas as pd
from dataset import SimpleDataset


def get_words(title: str) -> list:
    return title.split(' ')


def count_group_intersection_features(dataset: SimpleDataset) -> SimpleDataset:

    group_words = {}
    object_words = {}
    # get all words
    for idx, data in dataset.data.iterrows():
        group_id = data.group_id

        if group_id not in group_words:
            group_words[group_id] = set()

        object_words[idx] = set(get_words(data['title']))
        group_words[group_id] |= object_words[idx]

    object_stat = {}
    # compute num of intersections in group
    for idx, data in dataset.data.iterrows():
        group_id = data.group_id

        object_stat[idx] = len(object_words[idx]) / len(group_words[group_id])

    df = pd.DataFrame(object_stat.values(), columns=['count'])
    joined = dataset.data.assign(count=df['count'])
    joined['count'].fillna(0, inplace=True)

    group_avg_stat = joined.groupby(['group_id'])['count'].mean()
    group_avg_stat.name = "group_avg_count"
    joined = joined.join(group_avg_stat, on='group_id')
    dataset.data = joined

    return dataset
