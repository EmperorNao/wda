import sys
from os.path import join

from constants import LOCAL

from dataset import Dataset


if __name__ == "__main__":
    dataset = Dataset(
        path_to_data=join(LOCAL, "data/content/"),
        path_to_groups={
            'test': join(LOCAL, "data/test_groups.csv"),
            'train': join(LOCAL, "data/train_groups.csv"),
        },
        features_to_use=['title', 'keywords', 'abstract', 'description']
    )
    dataset.data.to_csv(join(LOCAL, "data/extended_dataset.csv"))
