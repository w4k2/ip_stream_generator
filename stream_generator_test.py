import strlearn as sl
from SemiSynthetic_StreamGenerator import SemiSynthetic_StreamGenerator
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB

X, y =  load_breast_cancer(return_X_y=True)

stream = SemiSynthetic_StreamGenerator(X, y, n_drifts=4, interpolation='cubic')

clf = GaussianNB()
evaluator = sl.evaluators.TestThenTrain()

evaluator.process(stream, clf)
drfs = stream._get_drifts()

import matplotlib.pyplot as plt

plt.plot(evaluator.scores[0])
plt.vlines(drfs, 0.5, 1, color='r')
plt.savefig('foo.png')

print(stream._get_drifts())