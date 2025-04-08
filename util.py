import collections
import math
import os
import random
import string

import config as cfg


def load_dataset(filename: str) -> dict:
    dataset = {}

    with open(filename, 'r') as f:
        for line in f:
            data = line.split()
            x = tuple(map(to_float, data[:-1]))
            y = data[-1]

            dataset[x] = y

    return dataset

def calc_dist(vec1: list, vec2: list) -> float:
    return sum(abs(vec1[i] - vec2[i]) for i in range(len(vec1)))

def to_float(x) -> float:
    return float(x.replace(',', '.'))

def shuffle_dict(d: dict) -> dict:
    result = {}

    while len(d) != 0:
        keys = list(d.keys())
        rand_index = random.randint(0, len(keys) - 1)
        result[keys[rand_index]] = d.pop(keys[rand_index])

    return result

def validate_letter(letter: str) -> bool:
    if letter in cfg.letters_alt.keys():
        letter = cfg.letters_alt[letter]

    return letter in string.ascii_lowercase

def text_to_vec(text: str) -> list[float]:
    letters = list(filter(lambda x: validate_letter(x), text.lower()))
    counter = collections.Counter(letters)
    vec = []

    for letter in string.ascii_lowercase:
        vec.append(counter[letter]/len(letters))
    return vec

def load_articles(filename: str, return_names: bool = False):
    dataset = {}
    names = os.listdir(filename)
    for name in names:
        filenames = os.listdir(os.path.join(filename, name))
        for i, file in enumerate(filenames):
            with open(os.path.join(filename, name, file), 'r') as f:
                dataset[tuple(text_to_vec(f.read()))] = name

    return dataset if not return_names else (names, dataset)

def get_vec_length(vec: list):
    sum = 0
    for x in vec:
        sum += x**2

    return math.sqrt(sum)

def get_normal_vec(vec: list) -> list:
    length = get_vec_length(vec)
    return list(map(lambda x: x/length, vec))