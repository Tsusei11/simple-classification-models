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

def categorize_vec(vec: list, categories: tuple) -> dict:
    sorted_vec = sorted(vec)
    step = len(sorted_vec)/len(categories)

    counter = 0
    current_category = categories[0]
    ranges = {current_category: [0]}
    for i, num in enumerate(sorted_vec):
        if counter == step:
            ranges[current_category].append(sorted_vec[i-1])
            current_category = categories[categories.index(current_category)+1]
            ranges[current_category] = [sorted_vec[i-1]]
            counter = 0

        vec[vec.index(num)] = current_category
        counter += 1
    ranges[current_category].append(math.inf)
    return ranges

def get_categorized_vec(vec: list, ranges: dict) -> None:
    for i, num in enumerate(vec):
        for category in ranges:
            if ranges[category][0] < num <= ranges[category][1]:
                vec[i] = category
                break

def get_categorized_attr(attr: list, ranges: list) -> None:
    for i, range_list in enumerate(ranges):
        for category in range_list:
            if range_list[category][0] < attr[i] <= range_list[category][1]:
                attr[i] = category
                break

def categorize_dataset(dataset: dict, categories: tuple) -> (dict, list):
    result = {}
    ranges = []

    attributes = get_attributes_vectors(dataset)
    for attr in attributes:
        ranges.append(categorize_vec(attr, categories))

    for col, v in enumerate(dataset.values()):
        attr_row = tuple([attributes[row][col] for row in range(len(attributes))])
        result[attr_row] = v

    return result, ranges

def get_categorized_dataset(dataset: dict, ranges: list) -> dict:
    result = {}

    attributes = get_attributes_vectors(dataset)
    for i, attr in enumerate(attributes):
        get_categorized_vec(attr, ranges[i])

    for col, v in enumerate(dataset.values()):
        attr_row = tuple([attributes[row][col] for row in range(len(attributes))])
        result[attr_row] = v

    return result

def get_attributes_vectors(dataset: dict) -> list:
    attributes_n = len(list(dataset.keys())[0])
    attributes = []

    for n in range(attributes_n):
        attributes.append([])

    for i, attr in enumerate(attributes):
        for k in dataset.keys():
            attr.append(k[i])

    return attributes

def draw_confusion_matrix(matrix: dict) -> None:
    print('\t\t\t\t\t', end='')
    for i in matrix.keys():
        print(i + ' (T)', end='\t')
    print()
    for i in matrix.keys():
        row = i + ' (P)' + '\t\t\t'
        for j in matrix[i].keys():
            row = row + str(matrix[i][j]) + '\t\t\t\t'
        print(row)