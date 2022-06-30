import numpy as np
import matplotlib.pyplot as plt

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
s = 'wisconsin.csv'

for chunk in range(200):

    fig, ax = plt.subplots(1,2,figsize=(15, int(0.8*methods)), sharey=True)

    for d_id, d in enumerate(interp):
        res = np.load('results/res_%s_%s.npy' % (s.split('.')[0], d))
        acc = np.load('results/acc_%s_%s.npy' % (s.split('.')[0], d))

        for i in range(methods+1):
            # print(res[i,:])
            x = np.argwhere(res[i,:]==2).flatten()
            x = x[x<chunk]
            # print(x)
            y = [i for k in range(len(x))]

            if i<methods:
                ax[d_id].scatter(x,y, c='gray')
                ax[d_id].plot((acc[i]+i)[:chunk], ls=":", color='cornflowerblue')
            else:
                ax[d_id].scatter(x, y, c='black')
        
        ax[d_id].vlines(idx, 0, methods, color='tomato', linewidth=1)
        # ax[d_id].vlines(chunk, 0, methods, color='black', linewidth=1)
        ax[d_id].set_xlim(0,200)
        ax[d_id].set_title("interpolation: %s" % d)
        ax[d_id].set_xticks(idx)
        ax[d_id].set_yticks(range(methods+1))
        ax[d_id].set_yticklabels(labels)
        ax[d_id].set_xlabel('chunk index')
        ax[d_id].spines.top.set_visible(False)
        ax[d_id].spines.right.set_visible(False)
        ax[d_id].grid()

    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('temp3/%i.png' % chunk)
    plt.clf()
    # exit()
