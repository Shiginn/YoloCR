from vstools import core, scale_value, vs

from .base import BaseCleaner


class YoloCRCleaner(BaseCleaner):
    thr_in: int
    thr_out: int
    rect_size: int
    # thr_sc_offset: float = 0,

    def __init__(self, thr_in: int = 220, thr_out: int = 70, rect_size: int = 3):
        """
        :param thr_in:              Binarization threshold of the subtitle inline. Higher means less errors but
                                    text might not be detected. Should not be higher than subtitle text luminosity.
                                    Defaults to 220 and ranges from 0 to 255 (will be scaled if clip is not 8-bits)

        :param thr_out:             Binarization threshold of the subtitle outline. Lower means more errors will be
                                    removed but text might be detected as error. Should not be lower than subtitle
                                    outline luminosity. Defaults to 70 and ranges from 0 to 255 (will be scaled if
                                    clip is not 8-bits)

        :param thr_sc_offset:       Offset the threshold of the subtitle timing detection. This threshold is determined
                                    based on detection box size and can be offset with this setting. Lower means more
                                    subtitles will be detected but might cause false positive. Defaults threshold is
                                    0.0035 when detection box is 1500x200 and goes down as detection box size
                                    increases. Threshold is between 0 and 1.

        :param rect_size:           Size of the rectangle used to detect cleaning errors. Higher means more errors
                                    will be removed but might detect text as error. Defaults to 8.
        """

        self.thr_in = thr_in
        self.thr_out = thr_out
        self.rect_size = rect_size

        # if not (isinstance(thr_in, int) and isinstance(thr_out, int)):
        #     raise ValueError("Binarization threshold must be integers.")

        # if self.clip.format.num_planes > 1:
        #     self.clip, *_ = core.std.SplitPlanes(self.clip)  # type: ignore

        # self.thr_in = self._scale_values(thr_in, self.clip.format.bits_per_sample)  # type: ignore[union-attr]
        # self.thr_out = self._scale_values(thr_out, self.clip.format.bits_per_sample)  # type: ignore[union-attr]

        # self.thr_sc_offset = thr_sc_offset

        # self.rect_size = rect_size

        super().__init__()

    def _clean(self, clip: vs.VideoNode):
        assert clip.format

        bnz_in = core.std.Binarize(clip, self.thr_in)
        bnz_out = core.std.Binarize(clip, self.thr_out)

        blank_clip = core.std.BlankClip(
            bnz_in, width=clip.width - self.rect_size * 2, height=clip.height - self.rect_size * 2, color=0
        )

        rect = core.std.AddBorders(
            blank_clip,
            left=self.rect_size,
            right=self.rect_size,
            top=self.rect_size,
            bottom=self.rect_size,
            color=scale_value(255, 8, clip.format.bits_per_sample),
        )

        overlap = core.std.Expr([rect, bnz_out], "x y min")

        ocr_issues = core.misc.Hysteresis(overlap, bnz_out)

        txt = core.std.MaskedMerge(bnz_in, core.std.BlankClip(bnz_in), ocr_issues)

        return txt.std.Maximum().std.Minimum().std.Invert().std.PlaneStats()
