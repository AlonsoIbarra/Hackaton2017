from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import random
import pickle
import numpy as np
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10_000_000


def create_lexicon(pos, neg):
    lexicon = []

    for filename in [pos, neg]:
        with open(filename, 'r') as f:
            contents = f.readlines()

            for line in contents[:hm_lines]:
                words = word_tokenize(line.lower())
                lexicon += list(words)

    lexicon = [lemmatizer.lemmatize(word) for word in lexicon]
    w_counts = Counter(lexicon)
    # w_counts = {'the': 42384, 'and': 123213, ... }

    l2 = []
    for keyword in w_counts:
        if 1000 > w_counts[keyword] > 50:
            l2.append(keyword)

    print(len(l2))

    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()

        for line in contents[:hm_lines]:
            words = word_tokenize(line.lower())
            words = [lemmatizer.lemmatize(word) for word in words]

            features = np.zeros(len(lexicon))

            for word in words:
                word = word.lower()

                if word in lexicon:
                    position = lexicon.index(word)
                    features[position] += 1

            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1, 0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size * len(features))
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(
        'pos.txt',
        'neg.txt')

    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
