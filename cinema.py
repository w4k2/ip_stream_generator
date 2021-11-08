import numpy as np
from numpy.random.mtrand import sample
from generator import ip_stream_generator
import os

def get_drifts(chunk_num, n_drifts):
    interval = chunk_num/n_drifts
    offset = 0.5*interval

    drfs = [interval*i for i in range(n_drifts)]
    return np.array(drfs) + offset

directory = 'datasets'
n_features = 8
chunk_n = 200
samples = chunk_n * 250
drifts_n = 10
d_types = ['nearest','cubic']

files = os.listdir(directory)

gt = get_drifts(chunk_n, drifts_n)
np.save('streams/drf_idx', gt)

for f in files:
    if f[0] == '.':
        continue

    ds = np.genfromtxt("%s/%s" % (directory, f), delimiter=",")
    X = ds[:, :-1]
    y = ds[:, -1].astype(int)

    for d in d_types:
        _X_s, _y = ip_stream_generator(X, y, 
                        total_samples=samples, 
                        stream_features=n_features, 
                        random_state=91872, 
                        n_drifts=drifts_n, 
                        interpolation=d)
    
        ds = np.concatenate((_X_s, _y[:, np.newaxis]), axis=1)
        np.save('streams/%s_%s' % (f.split('.')[0], d), ds)




    
