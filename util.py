import random


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