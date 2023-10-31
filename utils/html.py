import codecs
from bs4 import BeautifulSoup
from utils.text import lemmatize_text, join_lemmatized_doc


class HTMLParser:

    def parse_html(self, path_to_html: str) -> tuple[str, str]:
        with codecs.open(path_to_html, 'r', 'utf-8') as f:
            url = f.readline().strip()
            parsed = BeautifulSoup(f, 'lxml')

            title = parsed.title.text
            body = parsed.body.text

            return join_lemmatized_doc(lemmatize_text(title)), join_lemmatized_doc(lemmatize_text(body))
