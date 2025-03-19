from model import Model
from util import to_float
import config as cfg

k = int(input("Enter k: "))

model = Model(cfg.train, cfg.test)
k = len(model.train_data) if k > len(model.train_data) else abs(k)
attr_n = len(list(model.train_data.keys())[0])

count = model.test(k)

print(f'Correct test predictions: {count} ({count/len(model.test_data)*100:.2f}%)')

while input('Do you want to continue? (y/n): ').lower() == 'y':
    vec = list(map(to_float, input(f'Enter {attr_n}-dimensional vector: ').split()[:attr_n]))
    print(f'Prediction: {model.get_knn(vec, k)}')
