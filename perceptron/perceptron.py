import util


class Perceptron:

    def __init__(self, weights: list, threshold: float, learning_rate: float, desired):
        self.weights = weights
        self.weights.append(threshold)
        self.learning_rate = learning_rate
        self.desired = desired

    def compute_output(self, inputs: list) -> int:
        return 1 if self.compute_net(inputs) >= 0 else 0

    def compute_net(self, inputs: list) -> float:
        inputs = list(inputs)
        inputs.append(-1)
        return sum(w*x for w, x in zip(self.weights, inputs))

    def learn(self, training_set: dict) -> None:
        for inputs, decision in training_set.items():
            d = 1 if decision == self.desired else 0
            y = self.compute_output(inputs)

            while y != d:
                self.weights = [w + x for w, x in zip(self.weights, [x * (d-y) * self.learning_rate for x in inputs])]
                self.weights = util.get_normal_vec(self.weights)
                y = self.compute_output(inputs)

    def test(self, testing_set: dict) -> int:
        correct = 0
        for inputs, decision in testing_set.items():
            decision = 1 if decision == self.desired else 0

            if self.compute_output(inputs) == decision:
                correct += 1

        return correct
