import pandas as pd


class SimpleDataset:

    def __init__(self, path_to_titles: str, path_to_groups: str, train: bool):
        self.path_to_titles = path_to_titles
        self.path_to_groups = path_to_groups
        self.title_data = pd.read_csv(self.path_to_titles, sep='\t', encoding='utf-8', lineterminator='\n')
        self.group_data = pd.read_csv(self.path_to_groups)
        self.train = train

        self.group_data = self.group_data.join(self.title_data, on='doc_id', rsuffix="_titles").fillna({'title': ''})

    def __len__(self):
        return len(self.group_data)

    def __getitem__(self, item):
        return self.group_data.iloc[item]

    @property
    def data(self) -> pd.DataFrame:
        return self.group_data
