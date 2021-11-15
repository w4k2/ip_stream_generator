import numpy as np
import matplotlib.pyplot as plt
import os

directory = 'datasets'
datasets = os.listdir(directory)
print(datasets)

r_states = [123,3242,3241123,3,9182378,829,37]
methods = len(r_states)
labels=[]

for i in range(methods):
    labels.append('detector %i' % i)
labels = list(reversed(labels))
labels.append('ensemble')

idx = np.load('streams/drf_idx.npy')
print(idx)

interp = ['cubic', 'nearest']

for s in datasets:
    if s[0]=='.':
        continue
    print(s)

    fig, ax = plt.subplots(1,2,figsize=(13, int(0.7*methods)), sharey=True)

    for d_id, d in enumerate(interp):
        res = np.load('results/res_%s_%s.npy' % (s.split('.')[0], d))
        acc = np.load('results/acc_%s_%s.npy' % (s.split('.')[0], d))

        # print(res.shape)

        # print(res[-1,:])
        # x = np.argwhere(res[-1,:]==2).flatten()
        # print(x+1)

        # print(res[7,:])
        # x = np.argwhere(res[7,:]==2).flatten()
        # print(x+1)

        for i in range(methods+1):
            # print(res[i,:])
            x = np.argwhere(res[i,:]==2).flatten()
            # print(x)
            y = [i for k in range(len(x))]

            if i<methods:
                ax[d_id].scatter(x,y, c='gray')
                ax[d_id].plot(acc[i]+i, ls=":", color='cornflowerblue')
            else:
                ax[d_id].scatter(x,y, c='black')
        
        ax[d_id].vlines(idx, 0, methods, color='tomato')
        ax[d_id].set_xlim(0,200)
        ax[d_id].set_title("dataset: %s, interpolation: %s" % (s.split('.')[0], d))
        ax[d_id].set_xticks(idx)
        ax[d_id].set_yticks(range(methods+1))
        ax[d_id].set_yticklabels(labels)
        ax[d_id].set_xlabel('chunk index')
        ax[d_id].spines.top.set_visible(False)
        ax[d_id].spines.right.set_visible(False)
        ax[d_id].grid()

    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('figures/%s.eps' % s.split('.')[0])
    plt.clf()
    # exit()
