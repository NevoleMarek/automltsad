from abc import ABC, abstractmethod


class BaseDetector(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict_anomaly_scores(self):
        pass
