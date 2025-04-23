from collections import Counter

import util

class Classifier:
    def __init__(self, train_dataset):
        self.probs = None
        self.train(train_dataset)

    def train(self, train_dataset):
        self.probs = {}

        targets = list(train_dataset.values())
        attributes = util.get_attributes_vectors(train_dataset)

        targets_counter = Counter(targets)
        for target, count in targets_counter.items():
            self.probs[target] = []
            self.probs[target].append(count / len(targets))

        for target in set(targets):
            for i, attr in enumerate(attributes):
                categories = set(attr)
                attr_counter = Counter([attr[j] for j in range(len(attr)) if list(train_dataset.items())[j][1] == target])
                counter_sum = sum(attr_counter.values())
                distinct_n = len(categories)

                for category in attr_counter.keys():
                    attr_counter[category] = attr_counter[category] / counter_sum

                before = None
                for category in categories:
                    if category not in attr_counter:
                        before = attr_counter.copy() if before is None else before
                        attr_counter[category] = 1 / (counter_sum + distinct_n)

                if before is not None:
                    print(f'Before smoothing: {before}')
                    util.smooth(before, counter_sum, distinct_n)
                    for category in attr_counter.keys():
                        if category in before:
                            attr_counter[category] = before[category]
                    print(f'After smoothing: {attr_counter}\n')

                self.probs[target].append(attr_counter)

    def test(self, test_dataset) -> (int, dict):
        counter = 0
        confusion_matrix = {i: {j: 0 for j in self.probs.keys()} for i in self.probs.keys()}

        for attr, target in test_dataset.items():
            prediction = self.predict(attr)

            if prediction == target:
                counter += 1

            confusion_matrix[prediction][target] += 1


        return counter, confusion_matrix

    def predict(self, vec) -> str:
        predictions = {}
        for target in self.probs:
            prob = self.probs[target][0]
            for i in range(1, len(self.probs[target])):
                prob *= self.probs[target][i][vec[i - 1]]
            predictions[target] = prob

        return max(predictions, key=predictions.get)