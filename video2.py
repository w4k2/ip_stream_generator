import numpy as np
from generator2 import ip_stream_generator2
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import time

n_drfs = 4

ds = np.genfromtxt("datasets/banknote.csv", delimiter=",")
X = ds[:, :-1]
y_ = ds[:, -1].astype(int)

chunk_size = 250
chunk_number = 100

X_s_cubic, y, drfs = ip_stream_generator2(X, y_, 
                                        n_chunks=chunk_number, 
                                        chunk_size=chunk_size,
                                        stream_features=2,
                                        random_state=None, 
                                        n_drifts=n_drfs, 
                                        stabilize_factor=0.15,
                                        interpolation='linear')



drifts = drfs

interval = chunk_number/n_drfs

global_xmax = np.max(X_s_cubic[:,0])
global_xmin = np.min(X_s_cubic[:,0])

global_ymax = np.max(X_s_cubic[:,1])
global_ymin = np.min(X_s_cubic[:,1])


mean_chunk0_c=[]
mean_chunk1_c=[]

for i in range(1,chunk_number):
    mean_chunk0_c.append(np.mean(X_s_cubic[(i-1)*chunk_size:i*chunk_size,0]))
    mean_chunk1_c.append(np.mean(X_s_cubic[(i-1)*chunk_size:i*chunk_size,1]))
    
both_c = np.concatenate((np.array(mean_chunk0_c), np.array(mean_chunk1_c)))

ymin_plot = np.min(both_c)*1.1
ymax_plot = np.max(both_c)*1.1

for i in range(chunk_number):
    plt.clf()
    fig, ax = plt.subplots(1,2,figsize=(16*0.75,9*0.75))
    
    chunk = X_s_cubic[(i)*chunk_size:(i+1)*chunk_size]
    chunk_y = y[(i)*chunk_size:(i+1)*chunk_size]
    ax[0].scatter(chunk[:,0], chunk[:,1], c=['tomato' if chunk_y[i]==0 else 'cornflowerblue' for i in range(len(chunk_y))])
    ax[0].grid()
    # ax[0].set_title('chunk: %i' % i)
    ax[0].set_ylim(global_ymin, global_ymax)
    ax[0].set_xlim(global_xmin, global_xmax)
    ax[0].set_xlabel('feature A value')
    ax[0].set_ylabel('feature B value')
    ax[0].spines.top.set_visible(False)
    ax[0].spines.right.set_visible(False)
    
    
    ax[1].plot(np.arange(chunk_number-1) ,mean_chunk0_c, label = 'feature A', color='tomato')
    ax[1].plot(np.arange(chunk_number-1) ,mean_chunk1_c, label = 'feature B', color='cornflowerblue')
    ax[1].set_xlabel('chunk number')
    ax[1].set_ylabel('mean feature value')
    ax[1].grid()
    ax[1].vlines(drifts, ymin_plot, ymax_plot, ls=":", color='r')
    ax[1].vlines(i, ymin_plot, ymax_plot, ls="-", color='black')
    ax[1].set_ylim(ymin_plot, ymax_plot)
    ax[1].set_xlim(0, chunk_number)
    ax[1].legend()
    ax[1].spines.top.set_visible(False)
    ax[1].spines.right.set_visible(False)

    plt.tight_layout()
    time.sleep(1/24)
    plt.savefig("foo.png")
    plt.savefig("temp2/%i.png" % i)


