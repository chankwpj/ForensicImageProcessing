import abc

class Worker:
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def mask(self, im):
        "process image, then generate the mask"
        pass 