# Usage: python process_dataset.py dataset_dir output_dir

import sys
import os
import gzip
from xml.etree.cElementTree import iterparse,tostring
import traceback

from stanfordcorenlp import StanfordCoreNLP as corenlp
import tinysegmenter


# CoreNLP for tokenizing the translation output
CORENLP_DIR = "./stanford-corenlp-full-2018-02-27"
nlp = {}
for lang in ['en', 'de', 'fr']:
    nlp[lang] = corenlp(CORENLP_DIR, lang=lang)


def parse(itemfile):
    for event, elem in iterparse(itemfile):
        if elem.tag == "item":
            yield processItem(elem)
            elem.clear()


def processItem(item):
    """ Process a review.
    Implement custom code here. Use 'item.find('tagname').text' to access the properties of a review. 
    """
    review = {}
    # review.category = item.find("category").text
    review['rating'] = int(float(item.find("rating").text))
    # review.asin = item.find("asin").text
    # review.date = item.find("date").text
    review['text'] = item.find("text").text
    # review.summary = item.find("summary").text
    return review
    

def process_all_data(root_dir, out_dir):
    for lang in ['en', 'de', 'fr', 'ja']:
        for domain in ['books', 'dvd', 'music']:
            for split in ['train', 'test', 'unlabeled']:
                fn = os.path.join(root_dir, lang, domain, f'{split}.review')
                ofn = os.path.join(out_dir, lang, domain, f'{split}.tok.txt')
                with open(fn) as inf, open(ofn, 'w') as ouf:
                    print(f"Processing file: {fn}")
                    for review in parse(inf):
                        # binarize label
                        label = 1 if review['rating'] > 3 else 0
                        try:
                            # remove line breaks
                            raw_text = review['text'].replace('\n', ' ').replace('\t', ' ')
                            if lang == 'ja':
                                tok_text = tinysegmenter.tokenize(raw_text)
                            else:
                                tok_text = nlp[lang].word_tokenize(raw_text)
                        except:
                            print("Exception tokenizing", review)
                            continue
                        print(f"{label}\t{' '.join(tok_text)}", file=ouf)


if __name__ == "__main__":
    process_all_data(sys.argv[1], sys.argv[2])

