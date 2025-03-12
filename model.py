from util import *


class Model:

    def __init__(self, train_filename, test_filename, k):
        self.train_data = load_dataset(train_filename)
        self.test_data = load_dataset(test_filename)
        self.k = k

    def knn(self, vec):
        dist_dict = {}

        for x in self.train_data:
            dist_dict[x] = calc_dist(vec, x)

        dist_dict = sorted(dist_dict.items(), key=lambda pair: pair[1])[:self.k]

        count = {}
        for x in dist_dict:
            if self.train_data[x[0]] in count:
                count[self.train_data[x[0]]] += 1
            else:
                count[self.train_data[x[0]]] = 1

        return max(count, key=count.get)

    def test(self):
        count = 0

        for x, y in self.test_data.items():
            if self.knn(x) == y:
                count += 1

        return count


