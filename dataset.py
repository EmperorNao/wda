import pandas as pd


class SimpleDataset:

    def __init__(self, path_to_titles: str = None, path_to_groups: dict[str, str] = None):
        self.path_to_titles = path_to_titles
        self.path_to_groups = path_to_groups

        if path_to_titles and path_to_groups:
            self.train_group_data = pd.read_csv(path_to_groups['train'])
            self.train_group_data['IsTrain'] = 1
            self.test_group_data = pd.read_csv(path_to_groups['test'])
            self.test_group_data['IsTrain'] = 0

            self.group_data = pd.concat([self.train_group_data, self.test_group_data], ignore_index=True)

            self.group_data['title'] = ''
            with open(path_to_titles, encoding='utf-8') as f:
                for num_line, line in enumerate(f):
                    if num_line == 0:
                        continue
                    data = line.strip().split('\t', 1)
                    doc_id = int(data[0])
                    if len(data) == 1:
                        title = ''
                    else:
                        title = data[1]
                    if doc_id in self.group_data['doc_id']:
                        self.group_data.loc[self.group_data['doc_id'] == doc_id, 'title'] = title

    def from_csv(self, path_to_csv: str):
        self.group_data = pd.read_csv(path_to_csv)
        self.group_data['title'].fillna('', inplace=True)

    def __len__(self):
        return len(self.group_data)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.group_data.iloc[item]
        else:
            return self.group_data[item]

    @property
    def data(self) -> pd.DataFrame:
        return self.group_data

    @property
    def train_data(self) -> pd.DataFrame:
        return self.group_data[self.group_data['IsTrain'] == 1]

    @property
    def test_data(self) -> pd.DataFrame:
        return self.group_data[self.group_data['IsTrain'] == 0]

    @data.setter
    def data(self, value):
        self.group_data = value