import numpy as np
from scipy.interpolate import interp1d
import problexity as px
from scipy.spatial.distance import euclidean

def ip_stream_generator2(X, y,
                        n_chunks=200,
                        chunk_size=250,
                        stream_features = 2,
                        n_drifts = 3,
                        interpolation = 'nearest',
                        stabilize_factor = 0.15,
                        random_state = None,
                        binarize = True,
                        density = 155,
                        base_projection_pool_size=100,
                        eval_measures=[px.f1, px.n2]):
    np.random.seed(random_state)
    total_samples = n_chunks*chunk_size

    X_base = np.copy(X)
    y_base = np.copy(y)

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

    possible = np.linspace(0,n_samples,density)

    drift_basepoints = np.sort(np.random.choice(possible, n_drifts+1, replace=False).astype(int))
    drift_basepoints[0] = 0
    drift_basepoints[-1] = n_samples

    drfs=[]
    distances=[]

    [drfs.append(int(drift_basepoints[bp_id-1]+
            (drift_basepoints[bp_id]-drift_basepoints[bp_id-1])/2)) 
            for bp_id in range(1,len(drift_basepoints))]
    [distances.append(drift_basepoints[bp_id]-drift_basepoints[bp_id-1]) 
            for bp_id in range(1,len(drift_basepoints))]

    stream_basepoints = np.linspace(0, n_samples-1, n_samples).astype(int)
    base_projection_pool = np.random.normal(size=(base_projection_pool_size,
                                              concept_features,
                                              stream_features
                                              ))

    base = [m(X_base, y_base) for m in eval_measures]
    projection_scores = []
    for bp in base_projection_pool:
        X_temp = np.sum(X_base[:, :, np.newaxis] * bp, axis=1)

        projection_scores.append([m(X_temp, y_base) for m in eval_measures])
    
    dist = [euclidean(projection_scores[i], base) for i in range(base_projection_pool_size)]
    proba = -np.array(dist)
    proba -= np.min(proba)
    proba /=np.sum(proba)

    base_projections_idx = np.random.choice(range(base_projection_pool_size), p=proba, replace=False, size=n_drifts+1)
    base_projections = base_projection_pool[base_projections_idx]
                        
    _drift_basepoints = []
    _base_projections = []
    for p_id, p in enumerate(drift_basepoints):
            
        sep_neg = int(distances[p_id-1]*stabilize_factor)
        try:
            sep_pos = int(distances[p_id]*stabilize_factor)
        except:
            sep_pos = 0

        _drift_basepoints.append(p - sep_neg)
        _drift_basepoints.append(p)
        _drift_basepoints.append(p + sep_pos)
        
        [_base_projections.append(base_projections[p_id]) for i in range(3)]


    drift_basepoints = np.array(_drift_basepoints)[1:-1]
    base_projections = np.array(_base_projections)[1:-1]

    continous_projections = np.zeros((n_samples, concept_features, stream_features))
   
    for d_s in range(stream_features): 
        for d_c in range(concept_features): 
            original_values = base_projections[:, d_c, d_s]
            f = interp1d(drift_basepoints, original_values, kind=interpolation)
            # import matplotlib.pyplot as plt
            # import time
            # plt.plot(f(stream_basepoints))
            # plt.scatter(drift_basepoints, np.zeros((len(drift_basepoints))), c='r')
            # plt.savefig('foo2.png')
            # time.sleep(1/3)
            # exit()
            continous_projections[:, d_c, d_s] = f(stream_basepoints)
            
    X_s = np.sum(X[:, :, np.newaxis] * continous_projections, axis=1)

    return X_s, y, (np.array(drfs)/chunk_size).astype(int)
