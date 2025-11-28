from abc import ABC, abstractmethod

from vstools import vs


class AbstractCleaner(ABC):
    @abstractmethod
    def clean(self, clip: vs.VideoNode):
        pass
