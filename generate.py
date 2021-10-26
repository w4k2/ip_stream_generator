import numpy as np
import os
from dirty_dancing import ip_stream_generator

np.random.seed(505)

intepolations = ['cubic', 'nearest']
directory = 'datasets'
req_samples = 100000
features = 8
n_drifts = 5

replications=5

random_states=np.random.choice(np.arange(10,100000),5)
print(random_states)

for _,_,files in os.walk(directory):
    pass

for f in files:
    ds = np.genfromtxt("%s/%s" % (directory, f), delimiter=",")
    X = ds[:, :-1]
    y = ds[:, -1].astype(int)

    # brzydka binaryzacja
    y[y!=0] = 1

    imbalance = len(np.argwhere(y==0))/len(y)
    # print(imbalance)

    samples_0_cnt = int(imbalance*req_samples)
    samples_1_cnt = int(req_samples - samples_0_cnt)

    print(samples_0_cnt, samples_1_cnt)

    samples_0_cnt_missing = samples_0_cnt - len(np.argwhere(y==0))
    samples_1_cnt_missing = samples_1_cnt - len(np.argwhere(y==1))

    indexes_0 = np.random.choice(np.argwhere(y==0).flatten(), samples_0_cnt_missing)
    indexes_1 = np.random.choice(np.argwhere(y==1).flatten(), samples_1_cnt_missing)

    # print(indexes_0)
    # print(indexes_1)

    X_new0 = X[indexes_0]
    X_new1 = X[indexes_1]

    y_new0 = y[indexes_0]
    y_new1 = y[indexes_1]

    print(X.shape)
    X = np.concatenate((X, X_new0, X_new1))
    y = np.concatenate((y, y_new0, y_new1))

    p = np.random.permutation(len(y))

    X = X[p]
    y = y[p]

    for rep_id in range(replications):
        for i in intepolations:

            X_s, y = ip_stream_generator(X, y, features, n_drifts, i, random_states[rep_id])
            print(X_s.shape)
            print(y.shape)

            db = np.concatenate((X_s, y[:,np.newaxis]), axis=1)

            print(db.shape)

            np.save('streams/str_%s_%s_%i' % (f,i,random_states[rep_id]), db)
            
            # exit()










