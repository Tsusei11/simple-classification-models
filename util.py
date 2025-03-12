def load_dataset(filename):
    dataset = {}

    with open(filename, 'r') as f:
        for line in f:
            data = line.split()
            x = tuple(float(attr.replace(',', '.')) for attr in data[:-1])
            y = data[-1]

            dataset[x] = y

    return dataset

def calc_dist(vec1, vec2):
    return sum(abs(vec1[i] - vec2[i]) for i in range(len(vec1)))
