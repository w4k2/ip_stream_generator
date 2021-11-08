import numpy as np
from generator import ip_stream_generator

db = np.genfromtxt("datasets/wine.csv", delimiter=",")
X = db[:, :-1]
y_ = db[:, -1].astype(int)

X_s, y = ip_stream_generator(X, y_, 
                    total_samples=100000,
                    stream_features=10, 
                    random_state=1203, 
                    n_drifts=4, 
                    interpolation='cubic',
                    stabilize_factor=0.15)

ds = np.concatenate((X_s, y[:, np.newaxis]), axis=1)
np.save('stream', ds)

                

