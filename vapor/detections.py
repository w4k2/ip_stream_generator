import os
from sklearn.metrics import balanced_accuracy_score
import strlearn as sl
import numpy as np
from detectors.mlp_ensemble_meta import EMeta
from sklearn.base import clone

r_states = [123,3242,3241123,3,9182378,829,37]

base_method = EMeta(random_states=r_states)


directory = 'streams'
streams = os.listdir(directory)
n_chunks=200

for s in streams: 
    print(s)
  
    # if s not in ['breastcan_cubic.npy', 'breastcan_nearest.npy']:
    #     continue

    if s in ['drf_idx.npy']:
        continue

    stream = sl.streams.NPYParser('%s/%s' % (directory, s), chunk_size=250, n_chunks=n_chunks)
    clf = clone(base_method)

    chunk_acc = []
    for i in range(len(r_states)):
        chunk_acc.append([])

    while chunk := stream.get_chunk():
        if stream.chunk_id == n_chunks:
            break

        X, y = chunk
        y = y.astype(int)
        
        if stream.chunk_id==0:
            clf.partial_fit(X, y, np.unique(y))
        else:
            clf.predict(X)
            clf.partial_fit(X, y, np.unique(y))

            for i in range(len(r_states)):
                chunk_acc[i].append(balanced_accuracy_score(y, clf.pred[i]))

    dets = []
    for e in clf.ensemble:
        dets.append(e.detector.drift)
    dets.append(clf.drift)
    dets = np.array(dets)

    chunk_acc = np.array(chunk_acc)

    np.save("results/res_%s" % s.split('.')[0], dets)
    np.save("results/acc_%s" % s.split('.')[0], chunk_acc)

    # exit()
