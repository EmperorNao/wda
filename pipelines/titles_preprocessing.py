import sys
from os.path import join

from constants import LOCAL

from dataset import Dataset


if __name__ == "__main__":
    dataset = Dataset(
        path_to_titles=join(LOCAL, "data/docs_titles.tsv"),
        path_to_groups={
            'test': join(LOCAL, "data/test_groups.csv"),
            'train': join(LOCAL, "data/train_groups.csv"),
        }
    )
    dataset.data.to_csv(join(LOCAL, 'data/simple_data.csv'), index=None)
