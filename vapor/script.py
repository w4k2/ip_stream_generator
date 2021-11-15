from sklearn.datasets import make_classification
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

# Parameters
n_chunks = 200
chunk_size = 500
concept_features = 4
stream_features = 2
n_drifts = 5
interpolation = 'linear'
interpolation = 'nearest'
interpolation = 'cubic'
random_state = 1410

np.random.seed(random_state)

# Prepare base stream
n_samples = n_chunks * chunk_size
sconfig = {
    'n_samples': n_samples, 'n_informative':concept_features,
    'n_redundant':0, 'n_repeated': 0, 'n_features': concept_features,
    'random_state': random_state
}
X, y = make_classification(**sconfig)

# Prepare projections
drift_basepoints = np.linspace(0, n_samples, n_drifts+1).astype(int)
stream_basepoints = np.linspace(0, n_samples-1, n_samples).astype(int)
base_projections = np.random.normal(size=(len(drift_basepoints),
                                          concept_features,
                                          stream_features
                                          ))
#print(base_projections, base_projections.shape)
#normalizer = np.sum(base_projections, axis=(1))
#print(normalizer, normalizer.shape)

#base_projections *= normalizer[:, np.newaxis, :]

# Prepare continous projections
continous_projections = np.zeros((n_samples, concept_features, stream_features))
for d_s in range(stream_features):
    for d_c in range(concept_features):
        original_values = base_projections[:, d_c, d_s]
        f = interp1d(drift_basepoints, original_values, kind=interpolation)
        continous_projections[:, d_c, d_s] = f(stream_basepoints)

# Make projection
X_s = np.sum(X[:, :, np.newaxis] * continous_projections, axis=1)

print('X', X.shape)
print('drift_basepoints', drift_basepoints)
print('base_projections', base_projections.shape)
print('continous_projections', continous_projections.shape)
print('X_s', X_s.shape)

"""
Figure
"""
mask = np.random.uniform(size=len(y))<.05
for chunk_id in range(n_chunks):
    fig = plt.figure(figsize=(10,10))

    # Plot base_projections
    colors = ['blue', 'red']
    for d_s in range(stream_features):
        ax = plt.subplot(211)
        for d_c in range(concept_features):
            ax.scatter(drift_basepoints,
                       base_projections[:,d_c,d_s],
                       c=colors[d_s])
            ax.plot(drift_basepoints,
                    base_projections[:, d_c,d_s],
                    c=colors[d_s], ls=":")
            ax.plot(stream_basepoints,
                    continous_projections[:, d_c,d_s],
                    c=colors[d_s], ls="-")

        ax.set_ylim(-3,3)
        ax.grid(ls=":")
        ax.set_xlim(0, n_samples)

    # Plot chunk
    a = chunk_id*chunk_size
    b = (chunk_id+1) *chunk_size

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.vlines([chunk_id*chunk_size, (chunk_id+1) *chunk_size],
              -3, 3, color='black')

    chunk_X = X_s[a:b]
    chunk_y = y[a:b]

    print(chunk_X.shape)

    ax = plt.subplot(223)
    ax.scatter(chunk_X[:,0], chunk_X[:,1], c=chunk_y, cmap='bwr')
    #ax.set_xlim(np.min(X_s[:,0]),np.max(X_s[:,0]))
    #ax.set_ylim(np.min(X_s[:,1]),np.max(X_s[:,1]))

    #aa =
    #bb =
    ax.grid(ls=":")

    ax = plt.subplot(224)
    ss = X_s[mask]
    ax.scatter(ss[:,0], ss[:,1], c='#AAA')
    ax.scatter(chunk_X[:,0], chunk_X[:,1], c=chunk_y, cmap='bwr')
    ax.set_xlim(np.min(X_s[:,0]),np.max(X_s[:,0]))
    ax.set_ylim(np.min(X_s[:,1]),np.max(X_s[:,1]))
    ax.grid(ls=":")


    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('frames/%04i.png' % chunk_id)

    plt.close()
