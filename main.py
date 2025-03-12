from model import Model

k = int(input("Enter k: "))

model = Model('data/iris_training.txt', 'data/iris_test.txt', k)
attr_n = len(list(model.train_data.keys())[0])
count = model.test()
print(f'Correct test predictions: {count} ({count/len(model.test_data)*100:.2f}%)')

while True:
    choice = input('Do you want to continue? (y/n): ')
    if choice == 'n':
        break

    vec = tuple(float(attr.replace(',', '.')) for attr in input(f'Enter {attr_n}-dimensional vector: ').split()[:attr_n])

    print(f'Prediction: {model.knn(vec)}')