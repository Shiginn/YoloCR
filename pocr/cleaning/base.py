from abc import ABC, abstractmethod
from typing import Callable, Self

from vstools import vs


class BaseCleaner(ABC):
    """Base class for all cleaners."""

    postprocess_func: Callable[[vs.VideoNode], vs.VideoNode] | None = None

    @abstractmethod
    def _clean(self, clip: vs.VideoNode) -> vs.VideoNode:
        pass

    def with_postprocess(self, func: Callable[[vs.VideoNode], vs.VideoNode]) -> Self:
        """Adds a post-processing step to the cleaner.

        :param func:        Function that takes the cleaned output and returns a VideoNode.

        :return:            Same cleaner instance with postprocess function registered.
        """
        self.postprocess_func = func
        return self

    def run(self, clip: vs.VideoNode) -> vs.VideoNode:
        """Cleans the provided clip using the cleaner's algorithm.

        :param clip:        Clip to clean.

        :return:            Cleaned clip.
        """
        cleaned = self._clean(clip)
        return self.postprocess_func(cleaned) if self.postprocess_func else cleaned
