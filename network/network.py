import util
from perceptron.perceptron import Perceptron

class Network:
    def __init__(self, max_epochs: int, min_accuracy: float):
        self.layer = []
        self.max_epochs = max_epochs
        self.min_accuracy = min_accuracy

    def add_perceptron_list(self, perceptron: list[Perceptron]):
        self.layer.extend(perceptron)

    def train_layer(self, train_set, test_set) -> dict[str, float]:
        accuracies = {}
        for perceptron in self.layer:
            epoch = 1
            while perceptron.test(test_set) / len(test_set) < self.min_accuracy and epoch < self.max_epochs:
                perceptron.learn(train_set)
                epoch += 1

            accuracies[perceptron.desired] = perceptron.test(test_set) / len(test_set)

        return accuracies

    def predict(self, text: str):
        vec = util.text_to_vec(text)
        answer = {}
        for perceptron in self.layer:
            if perceptron.compute_output(vec) == 1:
                answer[perceptron.desired] = perceptron.compute_net(vec)

        if len(answer) == 1:
            return list(answer.keys())[0]
        elif len(answer) == 0:
            return None
        else:
            answer = dict(sorted(answer.items(), key=lambda item: item[1], reverse=True))
            return list(answer.keys())[0]