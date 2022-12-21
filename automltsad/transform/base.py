from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    '''
    Abstract base class for transformer classes.
    '''

    @abstractmethod
    def fit(self):
        '''
        Abstract method for fitting of a transformer.
        '''
        pass

    @abstractmethod
    def transform(self):
        '''
        Abstract method for transform functionality.
        '''
        pass
