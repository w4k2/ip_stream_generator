from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import balanced_accuracy_score

class Meta(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf, detector, reset = True, partial=False):
        self.base_clf = base_clf
        self.detector = detector
        self.reset = reset
        self.partial = partial

        self.clf = clone(base_clf)
        self.prev_pred = None
        self.acc = []


    def partial_fit(self, X, y, classes):
        # If first chunk or not resetting just partial fit & return
        if self.prev_pred is None or self.reset == False:
            [self.clf.partial_fit(X, y, classes) for i in range(100)]
            return self
        
        acc = balanced_accuracy_score(y, self.prev_pred)
        self.acc.append(acc)
        
        # Feed predicted, real to detector
        self.detector.feed(X, y, self.prev_pred)

        if len(self.detector.drift)==0:
            return self

        # If drift
        if self.detector.drift[-1]==2:
            # Reset clf
            self.clf = clone(self.base_clf)

            # Partial fit since drift if self.partial
            if self.partial and hasattr(self.detector, 'last_drf_idx'):
                self.clf.partial_fit(X[self.detector.last_drf_idx:], y[self.detector.last_drf_idx:], classes)
                return self
            # Else partial fit all chunk
            else:
                [self.clf.partial_fit(X, y, classes) for i in range(50)]

        return self

    def predict(self, X):
        self.prev_pred = self.clf.predict(X)
        return self.prev_pred
