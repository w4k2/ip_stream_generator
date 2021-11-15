import numpy as np
from generator import ip_stream_generator
import matplotlib.pyplot as plt

def get_drifts(chunk_num, n_drifts):
    interval = chunk_num/n_drifts
    offset = 0.5*interval

    drfs = [interval*i for i in range(n_drifts)]
    return np.array(drfs) + offset

n_drfs = 8

ds = np.genfromtxt("datasets/wisconsin.csv", delimiter=",")
X = ds[:, :-1]
y_ = ds[:, -1].astype(int)

X_s_cubic, y = ip_stream_generator(X, y_, total_samples=50000, stream_features=2, random_state=12783, n_drifts=n_drfs, interpolation='cubic')

chunk_size = 250
chunk_number = int(X_s_cubic.shape[0]/chunk_size)

drifts = get_drifts(chunk_number, n_drfs)

interval = chunk_number/n_drfs
basepoints = drifts + 0.5*interval
basepoints = basepoints.astype(int)

fig, ax = plt.subplots(2,4,figsize=(13,7))


ax = ax.ravel()

for i, bp in enumerate(basepoints):
    chunk = X_s_cubic[(bp-1)*chunk_size:bp*chunk_size]
    chunk_y = y[(bp-1)*chunk_size:bp*chunk_size]
    ax[i].scatter(chunk[:,0], chunk[:,1], c=['tomato' if chunk_y[i]==0 else 'cornflowerblue' for i in range(len(chunk_y))])
    ax[i].set_title('chunk %i' % bp)
    ax[i].set_ylabel('feature A value')
    ax[i].set_xlabel('feature B value')

    ax[i].spines.top.set_visible(False)
    ax[i].spines.right.set_visible(False)
    ax[i].grid()

print(basepoints)

plt.tight_layout()
plt.savefig("foo.png")
plt.savefig("figures/scatterplot.eps")


