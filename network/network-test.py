import random

from network import Network
import util
import config as cfg
from perceptron.perceptron import Perceptron

network = Network(100, 0.99)

langs, train_dataset, test_dataset = util.load_articles(cfg.articles)
perceptrones = []
for lang in langs:
    random_weights = [random.random() for _ in range(len(list(train_dataset.keys())[0]))]
    perceptrones.append(Perceptron(random_weights, random.random(), random.random(), lang))

network.add_perceptron_list(perceptrones)
network.train_layer(train_dataset, test_dataset)

print('Network trained.')

while input("Do you want to continue? [y/n]: ") == 'y':
    text = input("Enter text: ")
    print(network.predict(text))