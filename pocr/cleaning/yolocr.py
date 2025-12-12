from vsmasktools import Morpho
from vstools import core, scale_value, vs

from .base import BaseCleaner


class YoloCRCleaner(BaseCleaner):
    thr_fill: int
    thr_border: int
    expand_iter: int

    def __init__(self, thr_fill: int = 220, thr_border: int = 70, expand_iter: int = 2) -> None:
        """
        :param thr_in:              Binarization threshold of the subtitle fill. Higher means less errors but
                                    text might not be detected. Should not be higher than subtitle text luminosity.
                                    Defaults to 220 and ranges from 0 to 255 (will be scaled if clip is not 8-bits)

        :param thr_out:             Binarization threshold of the subtitle border. Lower means more errors will be
                                    removed but text might be detected as error. Should not be lower than subtitle
                                    border luminosity. Defaults to 70 and ranges from 0 to 255 (will be scaled if
                                    clip is not 8-bits)

        :param expand_iter:         Number of iterations for the expansion of the subtitle fill mask. Lower values will
                                    catch more errors but might remove parts of the subtitle. Defaults to 2.
        """
        super().__init__()

        self.thr_fill = thr_fill
        self.thr_border = thr_border
        self.expand_iter = expand_iter

    def _clean(self, clip) -> vs.VideoNode:
        bnz_fill = core.std.Binarize(clip=clip, threshold=scale_value(self.thr_fill, 8, clip.format))
        bnz_border = core.std.Binarize(clip=clip, threshold=scale_value(self.thr_border, 8, clip.format))

        bnz_fill_expand = Morpho().expand(bnz_fill, sw=self.expand_iter)

        diff = core.std.Expr([bnz_border, bnz_fill_expand], expr="x y - 0 max")
        diff_grow = core.misc.Hysteresis(diff, bnz_border)

        clean = core.std.Expr([bnz_fill, diff_grow], expr="x y - 0 max")

        return clean
