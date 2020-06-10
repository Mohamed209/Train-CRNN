import numpy as np
import wikipediaapi
import wikipedia
from tqdm import tqdm
import re


def pull_wikipedia_content(language, article_count=20):
    print("downloading {} {} random articles :)".format(
        article_count, language))
    _lengths = [1, 2, 3, 4, 5, 6, 7]
    _data_path = 'dataset/text_corpus/'
    lines = []
    if language == 'arabic':
        wiki_wiki = wikipediaapi.Wikipedia('ar')
        wikipedia.set_lang('ar')
    elif language == 'english':
        wiki_wiki = wikipediaapi.Wikipedia('en')
        wikipedia.set_lang('en')
    else:
        raise Exception('only supports arabic and english languages')
    for i in tqdm(range(article_count)):
        # choose random title
        title = wikipedia.random()
        # query wikipedia py title
        page = wiki_wiki.page(title).text
        with open(_data_path+'wiki_'+language+'.txt', 'a+', encoding='utf-8') as corpus:
            start = 0
            end = 1
            while start < len(page):
                end = np.random.choice(_lengths)
                ara_corpus.writelines(' '.join(page[start:start+end]))
                ara_corpus.writelines('\n')
                start += end


if __name__ == "__main__":
    pull_wikipedia_content(language='arabic')
    pull_wikipedia_content(language='english')
