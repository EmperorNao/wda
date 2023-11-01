import logging
from os.path import join

import pandas as pd
from tqdm import tqdm

from utils.html import HTMLParser
from utils.text import preprocess_text


class Dataset:

    def __init__(self,
                 path_to_groups: dict[str, str] = None,
                 path_to_titles: str = None,
                 path_to_data: str = None,
                 features_to_use: tuple[str] = ('title', )
                 ):
        self.path_to_groups = path_to_groups
        self.path_to_titles = path_to_titles
        self.path_to_data = path_to_data
        self.features_to_use = features_to_use

        if self.path_to_groups:
            self.train_group_data = pd.read_csv(path_to_groups['train'])
            self.train_group_data['IsTrain'] = 1
            self.test_group_data = pd.read_csv(path_to_groups['test'])
            self.test_group_data['IsTrain'] = 0

            self.group_data = pd.concat([self.train_group_data, self.test_group_data], ignore_index=True)

            if self.path_to_titles:
                self.group_data['title'] = ''
                with open(path_to_titles, encoding='utf-8') as f:
                    logging.info("Loading {titles}".format(titles=path_to_titles))
                    for num_line, line in tqdm(enumerate(f)):
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

            elif self.path_to_data:
                for feature in self.features_to_use:
                    self.group_data[feature] = ''

                parser = HTMLParser()
                logging.info(logging.INFO, "Loading HTML's {data}".format(data=path_to_data))
                for doc_id in tqdm(self.group_data['doc_id'].unique()):
                    html_file_path = join(self.path_to_data, "{doc_id}.dat".format(doc_id=doc_id))
                    ret = parser.parse_html(html_file_path, list(self.features_to_use))
                    for feature in self.features_to_use:
                        self.group_data.loc[self.group_data['doc_id'] == doc_id, feature] = \
                            preprocess_text(ret[feature])

    def from_csv(self, path_to_csv: str):
        self.group_data = pd.read_csv(path_to_csv)
        self.group_data['title'].fillna('', inplace=True)
        if 'body' in self.group_data.columns:
            self.group_data['body'].fillna('', inplace=True)

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
