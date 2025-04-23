import config as cfg
import util
from classifier import Classifier

train = util.load_dataset(cfg.train)
test = util.load_dataset(cfg.test)
categories = ('very small', 'small', 'medium', 'large', 'very large')
attr_n = len(list(train.keys())[0])

train, ranges = util.categorize_dataset(train, categories)
test = util.get_categorized_dataset(test, ranges)

classifier = Classifier(train)
correct, matrix = classifier.test(test)

print(f'Successfully tested classifier with {correct} correct predictions ({correct/len(test)*100:.2f}%)')

util.draw_confusion_matrix(matrix)

while input('Do you want to continue? (y/n): ').lower() == 'y':
    vec = list(map(util.to_float, input(f'Enter {attr_n}-dimensional vector: ').split()[:attr_n]))
    util.get_categorized_attr(vec, ranges)
    print(f'Prediction: {classifier.predict(vec)}')