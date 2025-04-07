import random
import util
from perceptron import Perceptron
import config as cfg

train_set = util.shuffle_dict(util.load_dataset(cfg.train))
test_set = util.load_dataset(cfg.test)
attr_n = len(list(train_set.keys())[0])
random_weights = [random.random() for _ in range(attr_n)]

perceptron = Perceptron(random_weights, random.random(), random.random(), cfg.perceptron_decision)
epochs = 0

while perceptron.test(test_set) / len(test_set) < 1:
    perceptron.learn(train_set)
    epochs += 1

correct = perceptron.test(test_set)
print(f'Perceptron successfully tested after {epochs} epochs with {correct} correct answers ({correct/len(test_set)*100:.2f}%)')

while input('Do you want to continue? (y/n): ').lower() == 'y':
    vec = list(map(util.to_float, input(f'Enter {attr_n}-dimensional vector: ').split()[:attr_n]))
    print(f'Prediction: {cfg.perceptron_decision if perceptron.compute_output(vec) == 1 else f"Not {cfg.perceptron_decision}"}')
