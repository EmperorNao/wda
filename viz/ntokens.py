from os.path import join

from dataset import Dataset

import numpy as np
import matplotlib.pyplot as plt

from constants import LOCAL


if __name__ == "__main__":
    dataset = Dataset()
    dataset.from_csv(join(LOCAL, "data/dataset.csv"))

    train_lens = []
    for idx, row in dataset.train_data.iterrows():
        train_lens.append(len(row['title'].split(' ')))

    test_lens = []
    for idx, row in dataset.test_data.iterrows():
        test_lens.append(len(row['title'].split(' ')))

    bins = np.linspace(0, 40, 40)

    plt.figure(figsize=(15, 10))
    plt.hist(train_lens, bins, density=False, label="train", alpha=0.5, color="red")  # density=False would make counts
    plt.hist(test_lens, bins, density=False, label="test", alpha=0.5, color="green")  # density=False would make counts
    plt.title("NToken distribution")
    plt.xlabel("Number of tokens")
    plt.ylabel("Number of sentences")
    plt.legend()
    plt.show()
