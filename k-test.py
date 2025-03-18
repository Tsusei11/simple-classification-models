from matplotlib import pyplot as plt

from model import Model
import seaborn as sns

model = Model('data/iris_training.txt', 'data/iris_test.txt')

precisions = [model.test(k) / len(model.test_data) for k in range(1, len(model.train_data) + 1)]
sns.lineplot(precisions)

plt.show()


