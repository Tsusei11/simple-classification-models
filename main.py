from model import Model
from util import to_float

k = int(input("Enter k: "))

model = Model('data/iris_training.txt', 'data/iris_test.txt')
k = len(model.train_data) if k > len(model.train_data) else abs(k)

attr_n = len(list(model.train_data.keys())[0])
count = model.test(k)
print(f'Correct test predictions: {count} ({count/len(model.test_data)*100:.2f}%)')

choice = input('Do you want to continue? (y/n): ').lower()
while choice == 'y':
    vec = list(map(to_float, input(f'Enter {attr_n}-dimensional vector: ').split()[:attr_n]))

    print(f'Prediction: {model.get_knn(vec, k)}')

    choice = input('Do you want to continue? (y/n): ').lower()
