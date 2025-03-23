import util


class Perceptron:

    def __init__(self, weights: list, threshold: float, learning_rate: float, right_decision):
        self.weights = weights
        self.weights.append(threshold)
        self.learning_rate = learning_rate
        self.right_decision = right_decision

    def compute(self, inputs: list) -> int:
        inputs = list(inputs)
        inputs.append(-1)
        return 1 if sum(w*x for w, x in zip(self.weights, inputs)) >= 0 else 0

    def learn(self, training_set: dict) -> None:
        for inputs, decision in training_set.items():
            d = 1 if decision == self.right_decision else 0
            y = self.compute(inputs)

            while y != d:
                self.weights = [w + x for w, x in zip(self.weights, [x * (d-y) * self.learning_rate for x in inputs])]
                y = self.compute(inputs)

    def test(self, testing_set: dict) -> int:
        correct = 0
        for inputs, decision in testing_set.items():
            decision = 1 if decision == self.right_decision else 0

            if self.compute(inputs) == decision:
                correct += 1

        return correct
