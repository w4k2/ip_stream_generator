from sklearn.base import BaseEstimator, ClassifierMixin, clone
from .mlp_meta import Meta
from sklearn.neural_network import MLPClassifier
from .ADWIN import ADWIN
import numpy as np

class EMeta(BaseEstimator, ClassifierMixin):
    def __init__(self, random_states, window=3, sensitivity = 0.5):
        self.random_states = random_states
        self.window = window
        self.sensitivity = sensitivity

        self.ensemble = [
            Meta(MLPClassifier(random_state=r, hidden_layer_sizes=(20)), ADWIN()) for r in random_states
        ]

        self.drift=[]
        self.cnt = 0
        self.pred=[]

    def partial_fit(self, X, y, classes):
        for e in self.ensemble:
            e.partial_fit(X, y, classes)
        
        self.cnt+=1
        if self.cnt==1:
            return self
        
        if len(self.drift) < self.window:
            self.drift.append(0)
        else:
            last_drfs = []
            for e in self.ensemble:
                last_drfs.append(e.detector.drift[-self.window:])
            
            last_drfs_sum = np.sum(last_drfs, axis=1)
            detected = last_drfs_sum[last_drfs_sum>0]

            if(len(detected)>=(self.sensitivity*len(self.random_states))):
                self.drift.append(2)
                # print("detected in id: ", len(self.drift))
                # print(last_drfs_sum)
            else:
                self.drift.append(0)

        return self

    def predict(self, X):

        pred = []
        for e in self.ensemble:
            pred.append(e.predict(X))
        
        self.pred = np.array(pred)
        mean_pred = np.mean(self.pred, axis=0)
        return np.rint(mean_pred)
