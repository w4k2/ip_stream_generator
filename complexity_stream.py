import numpy as np
from generator import ip_stream_generator
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import problexity as px

cmap = plt.cm.winter

def get_drifts(chunk_num, n_drifts):
    interval = chunk_num/n_drifts
    offset = 0.5*interval

    drfs = [interval*i for i in range(n_drifts)]
    return np.array(drfs) + offset

n_drfs = 4

ds = np.genfromtxt("datasets/wisconsin.csv", delimiter=",")
X = ds[:, :-1]
y_ = ds[:, -1].astype(int)

cc = px.ComplexityCalculator()
original_complexity = cc.fit(X, y_).complexity


X_s_cubic, y = ip_stream_generator(X, y_, total_samples=50000, stream_features=2, random_state=179234, n_drifts=n_drfs, interpolation='cubic')

chunk_size = 250
chunk_number = int(X_s_cubic.shape[0]/chunk_size)

drifts = get_drifts(chunk_number, n_drfs)

interval = chunk_number/n_drfs

global_xmax = np.max(X_s_cubic[:,0])
global_xmin = np.min(X_s_cubic[:,0])

global_ymax = np.max(X_s_cubic[:,1])
global_ymin = np.min(X_s_cubic[:,1])


mean_chunk0_c=[]
mean_chunk1_c=[]

all_complexity = np.full((chunk_number, 22), np.nan)

for i in range(1,chunk_number):
    mean_chunk0_c.append(np.mean(X_s_cubic[(i-1)*chunk_size:i*chunk_size,0]))
    mean_chunk1_c.append(np.mean(X_s_cubic[(i-1)*chunk_size:i*chunk_size,1]))
    
both_c = np.concatenate((np.array(mean_chunk0_c), np.array(mean_chunk1_c)))

ymin_plot = np.min(both_c)*1.1
ymax_plot = np.max(both_c)*1.1

for i in range(chunk_number):
    plt.clf()
    fig, ax = plt.subplots(2,4,figsize=(4*9*0.5,16*0.5))
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
    ax[1].vlines(drifts, ymin_plot, ymax_plot, ls=":", color='r', linewidth=0.5)
    ax[1].vlines(i, ymin_plot, ymax_plot, ls="-", color='black')
    ax[1].set_ylim(ymin_plot, ymax_plot)
    ax[1].set_xlim(0, chunk_number)
    ax[1].legend()

    cc = px.ComplexityCalculator()
    all_complexity[i] = cc.fit(chunk, chunk_y).complexity

    k=3

    # feature-based
    cols = cmap(np.linspace(0,1,5))
    for m_id, m in enumerate(cc.metrics[:5]):
        ax[2].plot(np.arange(chunk_number), medfilt(all_complexity[:,m_id], k), color=cols[m_id], label=m.__name__)
        ax[2].hlines(original_complexity[m_id], 0, chunk_number, color=cols[m_id], ls=":")
    ax[2].set_title("feature based")

    # linearity
    cols = cmap(np.linspace(0,1,3))
    for m_id, m in enumerate(cc.metrics[5:8]):
        ax[3].plot(np.arange(chunk_number), medfilt(all_complexity[:,m_id+5],k), color=cols[m_id], label=m.__name__)
        ax[3].hlines(original_complexity[m_id+5], 0, chunk_number, color=cols[m_id], ls=":")
    ax[3].set_title("linearity")

    # neighborhood
    cols = cmap(np.linspace(0,1,6))
    for m_id, m in enumerate(cc.metrics[8:14]):
        ax[4].plot(np.arange(chunk_number), medfilt(all_complexity[:,m_id+8],k), color=cols[m_id], label=m.__name__)
        ax[4].hlines(original_complexity[m_id+8], 0, chunk_number, color=cols[m_id], ls=":")
    ax[4].set_title("neighborhood")

    # network
    cols = cmap(np.linspace(0,1,3))
    for m_id, m in enumerate(cc.metrics[14:17]):
        ax[5].plot(np.arange(chunk_number), medfilt(all_complexity[:,m_id+14],k), color=cols[m_id], label=m.__name__)
        ax[5].hlines(original_complexity[m_id+14], 0, chunk_number, color=cols[m_id], ls=":")
    ax[5].set_title("network")

    # dimensionality
    cols = cmap(np.linspace(0,1,3))
    for m_id, m in enumerate(cc.metrics[17:20]):
        ax[6].plot(np.arange(chunk_number), medfilt(all_complexity[:,m_id+17],k), color=cols[m_id], label=m.__name__)
        ax[6].hlines(original_complexity[m_id+17], 0, chunk_number, color=cols[m_id], ls=":")
    ax[6].set_title("dinemsionality")

    # class imbalance
    cols = cmap(np.linspace(0,1,2))
    for m_id, m in enumerate(cc.metrics[20:]):
        ax[7].plot(np.arange(chunk_number), medfilt(all_complexity[:,m_id+20],k), color=cols[m_id], label=m.__name__)
        ax[7].hlines(original_complexity[m_id+20], 0, chunk_number, color=cols[m_id], ls=":")
    ax[7].set_title("class imbalance")

    for a in range(8):
        ax[a].spines.top.set_visible(False)
        ax[a].spines.right.set_visible(False)
        ax[a].grid(linewidth=0.5, linestyle=":")
    
    for a in range(2,8):
        ax[a].set_ylim(0,1)
        ax[a].legend(loc=2)
        ax[a].vlines(drifts, 0, 1, ls=":", color='r', linewidth=0.5)


    plt.tight_layout()
    plt.savefig("foo.png")
    plt.savefig("temp_c/c_%i.png" % i)
