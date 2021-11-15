import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

total_samples = 10000
stream_features = 1
n_drifts = 5
interpolations = ['nearest', 'linear', 'cubic']
stabilize_factor = 0.15

n_samples, concept_features = 50000, 1

_drift_basepoints = []
_base_projections = []

drift_basepoints = np.linspace(0, n_samples, n_drifts+1).astype(int) #bazowe miejsca koncepcji
stream_basepoints = np.linspace(0, n_samples-1, n_samples).astype(int) # przestrzen liniowa o dlugosci strumienia 
base_projections = np.random.normal(size=(len(drift_basepoints)))

# stabilizacja
stabilize = int(stabilize_factor*n_samples/n_drifts)
for p_id, p in enumerate(drift_basepoints):
    _drift_basepoints.append(p - stabilize)
    _drift_basepoints.append(p)
    _drift_basepoints.append(p + stabilize)
    
    [_base_projections.append(base_projections[p_id]) for i in range(3)]

drift_basepoints = np.array(_drift_basepoints)[1:-1]
base_projections = np.array(_base_projections)[1:-1]

continous_projections = np.zeros((n_samples))

fig, ax = plt.subplots(1, len(interpolations), figsize=(12,3), sharex = True, sharey=True)

for k_id, k in enumerate(interpolations):
    f = interp1d(drift_basepoints, base_projections, kind=k)
    continous_projections = f(stream_basepoints)

    ax[k_id].set_title(k)
    ax[k_id].set_xlabel('sample')
    ax[k_id].set_ylabel('projection value')
    ax[k_id].plot(continous_projections, c='cornflowerblue')
    ax[k_id].scatter(drift_basepoints, base_projections, color='tomato', marker='o', s=13)
    
    ax[k_id].spines.top.set_visible(False)
    ax[k_id].spines.right.set_visible(False)
    ax[k_id].grid()

plt.tight_layout()
plt.savefig('foo.png')
plt.savefig('figures/projections.eps')
exit()
