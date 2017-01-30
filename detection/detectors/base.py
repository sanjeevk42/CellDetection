import abc


class AbstractDetector(object):
    __metaclass__ = abc.ABCMeta

    def get_model(self):
        return self.model
