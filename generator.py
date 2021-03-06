import numpy as np
from scipy.interpolate import interp1d
    
def ip_stream_generator(X, y,
                        total_samples = 10000,
                        stream_features = 2,
                        n_drifts = 3,
                        interpolation = 'nearest',
                        stabilize_factor = 0.15,
                        random_state = None,
                        binarize = True):
    np.random.seed(random_state)

    if binarize:
        y[y!=0] = 1

    classes = np.unique(y)
    class_indexes =[]

    for c in classes:
        ir = len(np.argwhere(y==c))/len(y)
        samples = int(ir*total_samples)
        indexes = np.random.choice(np.argwhere(y==c).flatten(), samples)
        class_indexes.append(indexes)

    X_arrs = [X[ind] for ind in class_indexes]
    y_arrs = [y[ind] for ind in class_indexes]

    X = np.concatenate((X_arrs))
    y = np.concatenate((y_arrs))

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

    stabilize = int(stabilize_factor*n_samples/n_drifts)
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
            
            f = interp1d(drift_basepoints, original_values, kind=interpolation)
            continous_projections[:, d_c, d_s] = f(stream_basepoints)
            
    X_s = np.sum(X[:, :, np.newaxis] * continous_projections, axis=1)

    return X_s, y
