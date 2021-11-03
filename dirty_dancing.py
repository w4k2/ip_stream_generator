import numpy as np
from scipy.interpolate import interp1d

def ip_stream_generator(X, y,
                        total_samples = 10000,
                        stream_features = 2,
                        n_drifts = 3,
                        interpolation = 'nearest',
                        random_state = None):
    np.random.seed(random_state)

    # binaryzacja
    y[y!=0] = 1

    # resample
    imbalance = len(np.argwhere(y==0))/len(y)

    samples0 = int(imbalance*total_samples)
    indexes0 = np.random.choice(np.argwhere(y==0).flatten(), samples0)

    samples1 = int(total_samples-samples0)
    indexes1 = np.random.choice(np.argwhere(y==1).flatten(), samples1)

    X = np.concatenate((X[indexes0], X[indexes1]))
    y = np.concatenate((y[indexes0], y[indexes1]))

    p = np.random.permutation(len(y))

    X = X[p]
    y = y[p]

    n_samples, concept_features = X.shape

    _drift_basepoints =[]
    _base_projections=[]

    drift_basepoints = np.linspace(0, n_samples, n_drifts+1).astype(int)
    stream_basepoints = np.linspace(0, n_samples-1, n_samples).astype(int)
    base_projections = np.random.normal(size=(len(drift_basepoints),
                                              concept_features,
                                              stream_features
                                              ))

    # stabilizacja
    stabilize = int(0.15*n_samples/n_drifts)
    for p_id, p in enumerate(drift_basepoints):
        _drift_basepoints.append(p - stabilize)
        _drift_basepoints.append(p)
        _drift_basepoints.append(p + stabilize)
        
        [_base_projections.append(base_projections[p_id]) for i in range(3)]

    drift_basepoints = np.array(_drift_basepoints)[1:-1]
    base_projections = np.array(_base_projections)[1:-1]

    continous_projections = np.zeros((n_samples, concept_features, stream_features))
    for d_s in range(stream_features):
        for d_c in range(concept_features):
            original_values = base_projections[:, d_c, d_s]
            
            if (original_values.shape[0] <= 3) and (interpolation == 'cubic'):
                f_l = interp1d(drift_basepoints,
                               original_values,
                               kind='linear')
                u_drift_basepoints = np.linspace(0,n_samples,4).astype(int)
                u_original_values = f_l(u_drift_basepoints)

                f = interp1d(u_drift_basepoints,
                             u_original_values,
                             kind=interpolation)
                continous_projections[:, d_c, d_s] = f(stream_basepoints)
            else:
                f = interp1d(drift_basepoints, original_values, kind=interpolation)
                continous_projections[:, d_c, d_s] = f(stream_basepoints)

    X_s = np.sum(X[:, :, np.newaxis] * continous_projections, axis=1)

    return X_s, y
