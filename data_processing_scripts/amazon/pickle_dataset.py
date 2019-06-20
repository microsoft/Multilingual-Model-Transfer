import os
import pickle
import sys


def process_all_data(root_dir, out_dir):
    for lang in ['en', 'de', 'fr', 'ja']:
        for domain in ['books', 'dvd', 'music']:
            for split in ['train', 'test', 'unlabeled']:
                fn = os.path.join(out_dir, lang, domain, f'{split}.tok.txt')
                ofn = os.path.join(root_dir, lang, domain, f'{split}.pkl')
                with open(fn) as inf, open(ofn, 'wb') as ouf:
                    reviews = {'X': [], 'Y': []}
                    print(f"Processing file: {fn}")
                    for line in inf:
                        parts = line.split('\t')
                        x = parts[1].rstrip().split(' ')
                        y = int(parts[0])
                        reviews['X'].append(x)
                        reviews['Y'].append(y)
                    pickle.dump(reviews, ouf)


if __name__ == '__main__':
    process_all_data(sys.argv[1], sys.argv[2])
