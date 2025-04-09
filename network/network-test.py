import random

from network import Network
import util
import config as cfg
from perceptron.perceptron import Perceptron

network = Network(100, 1)

langs, train_dataset = util.load_articles(cfg.articles, True)
test_dataset = util.load_articles(cfg.articles_test, False)
perceptrones = []
for lang in langs:
    random_weights = [random.random() for _ in range(len(list(train_dataset.keys())[0]))]
    perceptrones.append(Perceptron(random_weights, random.random(), random.random(), lang))

network.add_perceptron_list(perceptrones)
accuracies = network.train_layer(train_dataset, test_dataset)

print('Network trained and tested.')
mean = 0
for lang, acc in accuracies.items():
    print(f'{lang} perceptron accuracy = {acc * 100:.2f}%')
    mean += acc

print(f'Mean network accuracy = {mean / len(langs)*100:.2f}%')

while input("Do you want to continue? [y/n]: ") == 'y':
    text = input("Enter text: ")
    print(network.predict(text))