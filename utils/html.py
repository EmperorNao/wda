import codecs
from bs4 import BeautifulSoup


class HTMLParser:

    def parse_html(self, path_to_html: str, parse_list: list[str]) -> dict[str, str]:
        with codecs.open(path_to_html, 'r', 'utf-8') as f:
            url = f.readline().strip()
            parsed = BeautifulSoup(f, 'lxml')

        ret = {}
        for parse in parse_list:
            ret[parse] = parsed.__getattr__(parse).text

        return ret
