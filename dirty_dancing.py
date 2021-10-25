import numpy as np
from scipy.interpolate import interp1d

def ip_stream_generator(X, y,
                        stream_features = 2,
                        n_drifts = 3,
                        interpolation = 'cubic',
                        random_state = None):
    np.random.seed(random_state)
    n_samples, concept_features = X.shape

    drift_basepoints = np.linspace(0, n_samples, n_drifts+1).astype(int)
    stream_basepoints = np.linspace(0, n_samples-1, n_samples).astype(int)
    base_projections = np.random.normal(size=(len(drift_basepoints),
                                              concept_features,
                                              stream_features
                                              ))

    continous_projections = np.zeros((n_samples, concept_features, stream_features))
    for d_s in range(stream_features):
        for d_c in range(concept_features):
            original_values = base_projections[:, d_c, d_s]
            f = interp1d(drift_basepoints, original_values, kind=interpolation)
            continous_projections[:, d_c, d_s] = f(stream_basepoints)

    X_s = np.sum(X[:, :, np.newaxis] * continous_projections, axis=1)

    return X_s, y
