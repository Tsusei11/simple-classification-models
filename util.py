def load_dataset(filename: str) -> dict:
    dataset = {}

    with open(filename, 'r') as f:
        for line in f:
            data = line.split()
            x = tuple(float(attr.replace(',', '.')) for attr in data[:-1])
            y = data[-1]

            dataset[x] = y

    return dataset

def calc_dist(vec1: list, vec2: list) -> float:
    return sum(abs(vec1[i] - vec2[i]) for i in range(len(vec1)))

def sort_dict_by_val(d: dict) -> dict:
    l = list(d.items())
    for i in range(len(l)-1):
        min_val = i
        for j in range(i, len(l)):
            if l[j][1] < l[min_val][1]:
                min_val = j

        temp = l[min_val]
        l[min_val] = l[i]
        l[i] = temp

    sorted_d = {}
    for k, v in l:
        sorted_d[k] = v

    return sorted_d

def to_float(x) -> float:
    return float(x.replace(',', '.'))
