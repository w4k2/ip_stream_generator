import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


np.random.seed(123)

total_samples = 10000
stream_features = 1
n_drifts = 5
interpolations = 'cubic'
stabilize_factor = [0.15, 0.25, 0.4]

fig, ax = plt.subplots(1, len(stabilize_factor), figsize=(12,3), sharex = True, sharey=True)

n_samples, concept_features = 50000, 1

drift_basepoints_base = np.linspace(0, n_samples, n_drifts+1).astype(int) #bazowe miejsca koncepcji
stream_basepoints_base = np.linspace(0, n_samples-1, n_samples).astype(int) # przestrzen liniowa o dlugosci strumienia 
base_projections_base = np.random.normal(size=(len(drift_basepoints_base)))

for id, st in enumerate(stabilize_factor):

    _drift_basepoints = []
    _base_projections = []

    # stabilizacja
    stabilize = int(st*n_samples/n_drifts)
    for p_id, p in enumerate(drift_basepoints_base):
        _drift_basepoints.append(p - stabilize)
        _drift_basepoints.append(p)
        _drift_basepoints.append(p + stabilize)
        
        [_base_projections.append(base_projections_base[p_id]) for i in range(3)]

    drift_basepoints = np.array(_drift_basepoints)[1:-1]
    base_projections = np.array(_base_projections)[1:-1]

    continous_projections = np.zeros((n_samples))


    f = interp1d(drift_basepoints, base_projections, kind=interpolations)
    continous_projections = f(stream_basepoints_base)

    ax[id].set_title("Stabilization factor: %.2f" % st)
    ax[id].set_xlabel('sample')
    ax[id].set_ylabel('projection value')
    ax[id].plot(continous_projections, c='cornflowerblue')
    ax[id].scatter(drift_basepoints, base_projections, color='tomato', marker='o', s=13)
    
    ax[id].spines.top.set_visible(False)
    ax[id].spines.right.set_visible(False)
    ax[id].grid()

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/stabilize.eps')
