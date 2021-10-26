import numpy as np
import strlearn as sl
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


n_chunks = 200
chunk_size = 500

stream = sl.streams.NPYParser('streams/str_shuttle-c2-vs-c4.csv_nearest_21629.npy',
                    n_chunks=n_chunks, chunk_size=chunk_size)

clf = MLPClassifier(hidden_layer_sizes=(20))

scores= []

while chunk := stream.get_chunk():
    if stream.chunk_id == n_chunks:
        break

    # Get current chunk
    X, y = chunk
    y = y.astype(int)

    # print(X.shape, y.shape)

    # Only train on first chunk
    if stream.chunk_id == 0:
        [clf.partial_fit(X, y, np.unique(y)) for i in range(10)]

    else:
        """
        Test
        """
        # Establish predictions
        y_pred = clf.predict(X)

        # Establish scores
        score = sl.metrics.balanced_accuracy_score(y, y_pred)

        scores.append(score)
        # print(len(scores))
        """
        Train
        """
        [clf.partial_fit(X, y) for i in range(10)]

plt.plot(scores)
plt.savefig("foo.png")