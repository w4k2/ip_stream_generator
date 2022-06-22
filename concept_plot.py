import numpy as np
from generator import ip_stream_generator
import matplotlib.pyplot as plt

def get_drifts(chunk_num, n_drifts):
    interval = chunk_num/n_drifts
    offset = 0.5*interval

    drfs = [interval*i for i in range(n_drifts)]
    return np.array(drfs) + offset

n_drfs = 3

ds = np.genfromtxt("datasets/sonar.csv", delimiter=",")
X = ds[:, :-1]
y_ = ds[:, -1].astype(int)

X_s_cubic, y = ip_stream_generator(X, y_, total_samples=50000, stream_features=2, random_state=2321, n_drifts=n_drfs, interpolation='cubic')
X_s_nearest, y = ip_stream_generator(X, y_, total_samples=50000, stream_features=2, random_state=2321, n_drifts=n_drfs, interpolation='nearest')

chunk_size = 250
chunk_number = int(X_s_cubic.shape[0]/chunk_size)

drifts = get_drifts(chunk_number, n_drfs)

mean_chunk0_c=[]
mean_chunk1_c=[]

mean_chunk0_n=[]
mean_chunk1_n=[]

x = np.arange(chunk_number)

for i in range(1,chunk_number):
    mean_chunk0_c.append(np.mean(X_s_cubic[(i-1)*chunk_size:i*chunk_size,0]))
    mean_chunk1_c.append(np.mean(X_s_cubic[(i-1)*chunk_size:i*chunk_size,1]))
    
    mean_chunk0_n.append(np.mean(X_s_nearest[(i-1)*chunk_size:i*chunk_size,0]))
    mean_chunk1_n.append(np.mean(X_s_nearest[(i-1)*chunk_size:i*chunk_size,1]))


both_c = np.concatenate((np.array(mean_chunk0_c), np.array(mean_chunk1_c)))
both_n = np.concatenate((np.array(mean_chunk0_n), np.array(mean_chunk1_n)))

fig, ax = plt.subplots(1, 2, figsize=(12,4), sharex=True, sharey=True)

ax[0].plot(np.arange(i) ,mean_chunk0_c, label = 'feature A', color='tomato')
ax[0].plot(np.arange(i) ,mean_chunk1_c, label = 'feature B', color='cornflowerblue')
ax[0].set_title("cubic")
ax[0].set_xlabel('chunk number')
ax[0].set_ylabel('mean feature value')
ax[0].vlines(drifts, np.min(both_c), np.max(both_c), ls=":", color='r')

ax[1].plot(np.arange(i) ,mean_chunk0_n, label = 'feature A', color='tomato')
ax[1].plot(np.arange(i) ,mean_chunk1_n, label = 'feature B', color='cornflowerblue')
ax[1].set_title("nearest")
ax[1].set_xlabel('chunk number')
# ax[1].set_ylabel('mean feature value')
ax[1].vlines(drifts, np.min(both_n), np.max(both_n), ls=":", color='r')

ax[0].legend()
ax[1].legend()
for i in range(2):
    ax[i].spines.top.set_visible(False)
    ax[i].spines.right.set_visible(False)
    ax[i].grid()

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/means.eps')
plt.savefig('figures/means.png')


