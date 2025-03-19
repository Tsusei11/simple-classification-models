from matplotlib import pyplot as plt

from model import Model
import seaborn as sns

model = Model('data/iris_training.txt', 'data/iris_test.txt')

K = [i for i in range(1, len(model.train_data) + 1)]
precisions = [model.test(k) / len(model.test_data) for k in K]
sns.lineplot(x=K, y=precisions)

plt.xlabel('k')
plt.ylabel('precision')
plt.grid(True)
plt.show()


