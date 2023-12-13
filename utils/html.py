import codecs
from bs4 import BeautifulSoup


class HTMLParser:

    def parse_html(self, path_to_html: str, parse_list: list[str]) -> dict[str, str]:
        with codecs.open(path_to_html, 'r', 'utf-8') as f:
            url = f.readline().strip()
            parsed = BeautifulSoup(f, 'lxml')

        ret = {}
        for parse in parse_list:
            if parse in ['keywords', 'abstract', 'description']:
                ret[parse] = ''
                # print(parsed.find_all('meta'))
                for meta in parsed.find_all('meta'):
                    if 'name' in meta.attrs:
                        if meta.attrs['name'] == parse:
                            if 'content' in meta.attrs:
                                ret[parse] += meta.attrs['content'] + ' '
            else:
                ret[parse] = parsed.__getattr__(parse).text

        return ret