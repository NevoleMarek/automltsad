from abc import ABC, abstractmethod


class BaseDetector(ABC):
    '''
    Base abstract class for detectors.
    '''

    @abstractmethod
    def fit(self):
        '''
        Abstract method for fitting of a detector.
        '''
        pass

    @abstractmethod
    def predict(self):
        '''
        Abstract method for predictions of a detector.

        Should return labels.
        '''
        pass

    @abstractmethod
    def predict_anomaly_scores(self):
        '''
        Abstract method for predictions of anomaly scores of a detector.

        Should return anomaly scores.
        '''
        pass
