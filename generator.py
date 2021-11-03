import numpy as np
from scipy.interpolate import interp1d
import os

def generate(directory = 'datasets',
              drift_types = ['cubic'], 
              total_samples = 100000,
              stream_features = 5,
              n_drifts = 3,
              replications = 1,
              random_state = None):
    
    np.random.seed(random_state)
    intepolations = drift_types
    req_samples = total_samples

    random_states=np.random.choice(np.arange(10,100000),replications)
    print(random_states)

    for _,_,files in os.walk(directory):
        pass

    for f in files:
        f_ = f.split('.')[0]
        print(f_)
        ds = np.genfromtxt("%s/%s" % (directory, f), delimiter=",")
        X = ds[:, :-1]
        y = ds[:, -1].astype(int)

        # binaryzacja
        y[y!=0] = 1

        samples_0_cnt = int(len(np.argwhere(y==0))/len(y)*req_samples)
        samples_1_cnt = int(req_samples - samples_0_cnt)

        samples_0_cnt_missing = samples_0_cnt - len(np.argwhere(y==0))
        samples_1_cnt_missing = samples_1_cnt - len(np.argwhere(y==1))

        indexes_0 = np.random.choice(np.argwhere(y==0).flatten(), samples_0_cnt_missing)
        indexes_1 = np.random.choice(np.argwhere(y==1).flatten(), samples_1_cnt_missing)

        X = np.concatenate((X, X[indexes_0], X[indexes_1]))
        y = np.concatenate((y, y[indexes_0], y[indexes_1]))

        p = np.random.permutation(len(y))

        X = X[p]
        y = y[p]

        n_samples, concept_features = X.shape

        for rep_id in range(replications):
            for i in intepolations:

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

                        if (original_values.shape[0] <= 3) and (i == 'cubic'):
                            f_l = interp1d(drift_basepoints,
                                        original_values,
                                        kind='linear')
                            u_drift_basepoints = np.linspace(0,n_samples,4).astype(int)
                            u_original_values = f_l(u_drift_basepoints)

                            f = interp1d(u_drift_basepoints,
                                        u_original_values,
                                        kind=i)
                            continous_projections[:, d_c, d_s] = f(stream_basepoints)
                        else:
                            f = interp1d(drift_basepoints, original_values, kind=i)
                            continous_projections[:, d_c, d_s] = f(stream_basepoints)

                X_s = np.sum(X[:, :, np.newaxis] * continous_projections, axis=1)

                db = np.concatenate((X_s, y[:,np.newaxis]), axis=1)
                
                np.save('streams/stream_%s_%s_%i' % (f_,i,random_states[rep_id]), db)
            


generate(directory = 'datasets',
         drift_types = ['cubic'], 
         total_samples = 1000,
         stream_features = 5,
         n_drifts = 3,
         replications = 1,
         random_state = 1291)