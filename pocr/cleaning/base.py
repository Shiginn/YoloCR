from abc import ABC, abstractmethod
from typing import Callable, Self

from vstools import core, vs


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

    def run(self, clip: vs.VideoNode, show_mask: bool = False) -> vs.VideoNode:
        """Cleans the provided clip using the cleaner's algorithm.

        :param clip:        Clip to clean.
        :param show_mask:   If true, returns the mask instead of the cleaned clip. Defaults to False.

        :return:            Cleaned clip.
        """
        subs_mask = self._clean(clip)
        if self.postprocess_func:
            subs_mask = self.postprocess_func(subs_mask)

        if show_mask:
            return subs_mask

        blank = core.std.BlankClip(clip)
        return core.std.MaskedMerge(blank, clip, subs_mask)
