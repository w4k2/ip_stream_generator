import numpy as np
from generator2 import ip_stream_generator2
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import problexity as px

cmap = plt.cm.winter
n_drfs = 4

ds = np.genfromtxt("datasets/wisconsin.csv", delimiter=",")
X = ds[:, :-1]
y_ = ds[:, -1].astype(int)

measures = [px.f1, px.n2]
original_complexity = [m(X, y_) for m in measures]

chunk_size = 250
chunk_number = 100

X_s_cubic, y, drifts = ip_stream_generator2(X, y_, chunk_size=chunk_size, n_chunks=chunk_number, stream_features=2, random_state=179234, n_drifts=n_drfs, interpolation='cubic')

interval = chunk_number/n_drfs

global_xmax = np.max(X_s_cubic[:,0])
global_xmin = np.min(X_s_cubic[:,0])

global_ymax = np.max(X_s_cubic[:,1])
global_ymin = np.min(X_s_cubic[:,1])

mean_chunk0_c=[]
mean_chunk1_c=[]

all_complexity = np.full((chunk_number, 2), np.nan)

for i in range(1,chunk_number):
    mean_chunk0_c.append(np.mean(X_s_cubic[(i-1)*chunk_size:i*chunk_size,0]))
    mean_chunk1_c.append(np.mean(X_s_cubic[(i-1)*chunk_size:i*chunk_size,1]))
    
both_c = np.concatenate((np.array(mean_chunk0_c), np.array(mean_chunk1_c)))

ymin_plot = np.min(both_c)*1.1
ymax_plot = np.max(both_c)*1.1

for i in range(chunk_number):
    plt.clf()
    fig, ax = plt.subplots(2,2,figsize=(16,9))
    ax = ax.ravel()
    
    chunk = X_s_cubic[(i)*chunk_size:(i+1)*chunk_size]
    chunk_y = y[(i)*chunk_size:(i+1)*chunk_size]
    ax[0].scatter(chunk[:,0], chunk[:,1], c=['tomato' if chunk_y[i]==0 else 'cornflowerblue' for i in range(len(chunk_y))])
    ax[0].set_ylim(global_ymin, global_ymax)
    ax[0].set_xlim(global_xmin, global_xmax)
    ax[0].set_xlabel('feature A value')
    ax[0].set_ylabel('feature B value')

    ax[1].plot(np.arange(chunk_number-1) ,mean_chunk0_c, label = 'feature A', color='tomato')
    ax[1].plot(np.arange(chunk_number-1) ,mean_chunk1_c, label = 'feature B', color='cornflowerblue')
    ax[1].set_xlabel('chunk number')
    ax[1].set_ylabel('mean feature value')
    ax[1].vlines(drifts, ymin_plot, ymax_plot, ls=":", color='r', linewidth=1.5)
    ax[1].vlines(i, ymin_plot, ymax_plot, ls="-", color='black')
    ax[1].set_ylim(ymin_plot, ymax_plot)
    ax[1].set_xlim(0, chunk_number)
    ax[1].legend()

    all_complexity[i] = [m(chunk, chunk_y) for m in measures]

    k=3

    # f1
    ax[2].plot(np.arange(chunk_number), medfilt(all_complexity[:,0], k), color='r', label='F1')
    ax[2].hlines(original_complexity[0], 0, chunk_number, color='r', ls=":")
    ax[2].set_title("F1")
    ax[2].legend(loc=2)
    ax[2].vlines(drifts, 0, 1, ls=":", color='r', linewidth=1.5)
    ax[2].set_ylim(0,1)

    # n2
    ax[3].plot(np.arange(chunk_number), medfilt(all_complexity[:,1],k), color='r', label='N2')
    ax[3].hlines(original_complexity[1], 0, chunk_number, color='r', ls=":")
    ax[3].set_title("N2")
    ax[3].legend(loc=2)
    ax[3].vlines(drifts, 0, 1, ls=":", color='r', linewidth=1.5)
    ax[3].set_ylim(0,1)

    for a in range(4):
        ax[a].spines.top.set_visible(False)
        ax[a].spines.right.set_visible(False)
        ax[a].grid(linewidth=0.5, linestyle=":")

    plt.tight_layout()
    plt.savefig("foo.png")
    plt.savefig("temp_c/%0*d.png" % (4, i))
