from util import *


class ModelKNN:

    def __init__(self, train_filename: str, test_filename: str):
        self.train_data = load_dataset(train_filename)
        self.test_data = load_dataset(test_filename)

    def get_knn(self, vec: list, k: int) -> str:
        dist_dict = {}

        for x in self.train_data:
            dist_dict[x] = calc_dist(vec, x)

        nearest_neighbours = list(dict(sorted(dist_dict.items(), key=lambda pair: pair[1])).keys())[:k]

        return self.train_data[max(nearest_neighbours, key=nearest_neighbours.count)]

    def test(self, k: int) -> int:
        count = 0

        for x, y in self.test_data.items():
            if self.get_knn(x, k) == y:
                count += 1

        return count
