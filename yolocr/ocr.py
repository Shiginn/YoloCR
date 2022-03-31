import html
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytesseract
import vapoursynth as vs

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from fractions import Fraction
from shutil import rmtree
from typing import Tuple, Optional, List

from PIL import Image
from pytimeconv import Convert

core = vs.core


class YoloCR():
    """OCR Class"""

    def __init__(
        self,
        clip_hardsub: vs.VideoNode, /, *,
        coords: Tuple[int, int, int],
        coords_alt: Optional[Tuple[int, int, int]] = None,
        thr_in: int = 220,
        thr_out: int = 70,
        thr_sc_offset: float = 0,
        rect_size: int = 8
    ) -> None:
        """
        :param clip_hardsub:        Hardsubbed clip to OCR.

        :param coords:              Postion of the bottom detection box (width, height, vertical margin from the bottom).

        :param coords_alt:          Postion of the top detection box (width, height, vertical margin from the top).
                                    Will increase processing time.

        :param thr_in:              Binarization threshold of the subtitle inline.
                                    Higher means less errors but text might not be detected.
                                    Should not be higher than subtitle text luminosity.
                                    Defaults to 220 and ranges from 0 to 255 (will be scaled if clip is not 8-bits)

        :param thr_out:             Binarization threshold of the subtitle outline.
                                    Lower means more errors will be removed but text might be detected as error.
                                    Should not be lower than subtitle outline luminosity.
                                    Defaults to 70 and ranges from 0 to 255 (will be scaled if clip is not 8-bits)

        :param thr_sc_offset:       Offset the threshold of the subtitle timing detection.
                                    This threshold is determined based on detection box size and can be offset with this setting.
                                    Lower means more subtitles will be detected but might cause false positive.
                                    Defaults threshold is 0.0035 when detection box is 1500x200 and goes down as detection box size increases.
                                    Threshold is between 0 and 1.

        :param rect_size:           Size of the rectangle used to detect cleaning errors.
                                    Higher means more errors will be removed but might detect text as error.
                                    Defaults to 8.
        """
        self.clip = clip_hardsub

        if self.clip.format is None:
            raise ValueError("Variable format clip are not supported.")

        if self.clip.format.color_family not in [vs.GRAY, vs.YUV]:
            raise ValueError("Input clip must be GRAY or YUV.")

        if not (isinstance(thr_in, int) and isinstance(thr_out, int)):
            raise ValueError("Binarization threshold must be integers.")

        if self.clip.format.num_planes > 1:
            self.clip, *_ = core.std.SplitPlanes(self.clip)  # type: ignore

        self.thr_in = self._scale_values(thr_in, self.clip.format.bits_per_sample)  # type: ignore[union-attr]
        self.thr_out = self._scale_values(thr_out, self.clip.format.bits_per_sample)  # type: ignore[union-attr]

        self.thr_sc_offset = thr_sc_offset

        self.coords = self._convert_coords(self.clip, coords)
        self.coords_alt = self._convert_coords(self.clip, coords_alt, True) if coords_alt else None

        self.rect_size = rect_size


    def extract_frames(self) -> None:
        """Extract the subtitles into images ready for OCR"""

        try:
            os.mkdir("filtered_images")
        except FileExistsError:
            rmtree("filtered_images")
            os.mkdir("filtered_images")

        self._write_sub_frames(
            self._cleaning(core.std.Crop(self.clip, *self.coords))
        )

        if self.coords_alt:
            self._write_sub_frames(
                self._cleaning(core.std.Crop(self.clip, *self.coords_alt)),
                alt=True
            )


    def write_subs(self, lang: str, output_file: str = "subs.ass") -> None:
        """Write the ASS file from the extracted frames

        :param lang:        Language of the input subtitles.
        """

        if lang not in pytesseract.get_languages():
            raise ValueError(f"Language '{lang}' is not installed")

        print("OCRing images...")

        file_to_process = os.listdir("filtered_images")
        file_to_process = [file for file in file_to_process if file.endswith("_ocr.png")]

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            lines = executor.map(
                partial(
                    self._ocr_image,
                    lang=lang,
                    fps=Fraction(self.clip.fps_num, self.clip.fps_den)
                ), file_to_process
            )

        print("Writing ASS file...")

        with open(output_file, "w", encoding="utf8") as sub_file:
            sub_file.write(self._get_sub_headers(self.clip.width, self.clip.height))
            sub_file.write("\n".join(sorted(lines)))


    def _cleaning(self, clip: vs.VideoNode) -> vs.VideoNode:
        """Prepares the clip for the OCR by removing everything that is not subtitles.

        :param clip:        Clip to process.

        :returns:           Cleaned clip.
        """
        assert clip.format

        bnz_in = core.std.Binarize(clip, self.thr_in)
        bnz_out = core.std.Binarize(clip, self.thr_out)

        blank_clip = core.std.BlankClip(
            bnz_in,
            width=clip.width - self.rect_size * 2,
            height=clip.height - self.rect_size * 2,
            color=0
        )

        rect = core.std.AddBorders(
            blank_clip,
            left=self.rect_size,
            right=self.rect_size,
            top=self.rect_size,
            bottom=self.rect_size,
            color=self._scale_values(255, clip.format.bits_per_sample)
        )

        overlap = core.std.Expr([rect, bnz_out], "x y min")

        ocr_issues = core.misc.Hysteresis(overlap, bnz_out)

        txt = core.std.MaskedMerge(
            core.std.BlankClip(bnz_in),
            bnz_in,
            ocr_issues.std.Invert()
        )

        return txt.std.Maximum().std.Minimum().std.Minimum().std.Maximum()


    def _write_sub_frames(self, clip: vs.VideoNode, alt: bool = False) -> None:
        """Write images with subtitles from processed clip

        :param clip:        Cleaned clip to extract frames from.
        :param alt:         Whether or not to use alt coords. Defaults to False
        """
        scene_changes = [0, clip.num_frames - 1]

        thr_sc = 0.0035 * (300000 / (clip.width * clip.height)) + self.thr_sc_offset

        def _get_frame_ranges(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
            if f.props["_SceneChangePrev"] == 1 or f.props["_SceneChangeNext"] == 1:
                scene_changes.insert(-1, n)
            return clip

        ocr = core.std.FrameEval(
            clip,
            partial(_get_frame_ranges, clip=clip),
            prop_src=clip.misc.SCDetect(thr_sc)
        )

        print(f"Finding subtitles in {'main' if not alt else 'alt'} clip...")

        with open(os.devnull, "wb") as devnull:
            ocr.output(
                devnull,
                progress_update=lambda x, y: print("%d/%d [%.1f%%]" % (x, y, x / y * 100), end="\r")
            )

        scene_changes = sorted(scene_changes)
        txt_scenes = [(scene_changes[i], scene_changes[i + 1]) for i in range(0, len(scene_changes) - 1, 2)][1::2]

        print("\nWriting image files...\n")

        clip = core.std.Invert(clip).resize.Bicubic(format=vs.RGB24, matrix_in=1)

        for (start_f, end_f) in txt_scenes:
            path = f"{os.getcwd()}/filtered_images/{start_f}_{end_f}{'_alt' if alt else ''}_ocr.png"

            with clip.get_frame(start_f) as f:
                f_array = np.dstack(f)  # type: ignore
                img = Image.fromarray(f_array, mode="RGB")
                img.save(path)


    @staticmethod
    def _ocr_image(file: str, lang: str, fps: Fraction) -> str:
        """OCR a single image and returns the result in ASS format

        :param file:        Path to the image to process (format must be : <start timestamp>_<end_timestamp><_alt (optional)>.ext).
        :param lang:        Language of the input text.

        :returns:           OCR'd text in ASS format.
        """
        tesseract_hocr: bytes = pytesseract.image_to_pdf_or_hocr(
            f"filtered_images/{file}",
            lang,
            config="--oem 0 --psm 6",
            extension="hocr"
        )

        txt = tesseract_hocr.decode("utf8")

        txt = txt.replace("<em>", "{\\i1}") \
                 .replace("</em>", "{\\i0}")        # replace html italics with ASS italics

        txt = "".join(
            list(ET.fromstring(txt).itertext())     # extract text from xml
        ).replace("{\\i1} {\\i0}", "").strip()      # remove whitespaces before/after text

        txt = html.unescape(txt)                    # unescape html caracters
        txt = re.sub(r"\n{1} +", " ", txt)          # convert 1 line break into whitespace
        txt = re.sub(r" {2,}", r" \\N", txt)        # convert 2+ line break into one ASS linebreak
        txt = re.sub(                               # remove redundant italic tags
            r"{\\i0}(\W+| \\N){\\i1}",
            lambda x: x.group()[5:-5],
            txt
        )

        corrections: List[Tuple[str, str]] = [
            ("’", "'"), ("‘", "'"), (" '", "'"), ("!'", "l'"), ("I'", "l'"),
            ("_", "-"), ('—', '-'), ("...", "…"),
            ("<<", "«"), (">>", "»"), ("« ", "\""), (" »", "\""),
            ('II', 'Il'), ("iI", "il"),
            ("{\\i1}-{\\i0}", "-")
        ]

        for correction in corrections:
            txt = txt.replace(*correction)

        # add \an8 tag for alt track
        if "_alt" in file:
            if txt.startswith("{"):
                txt = txt[:1] + r"\an8" + txt[1:]
            else:
                txt = r"{\an8}" + txt

        start_f, end_f, *_ = file.split("_")

        start_ts = Convert.f2assts(int(start_f), fps)
        end_ts = Convert.f2assts(int(end_f) + 1, fps)

        return f'Dialogue: 10,{start_ts},{end_ts},Default,,0,0,0,,{txt}'


    @property
    def preview_coords(self) -> vs.VideoNode:
        """Preview of the OCR zone(s)"""
        base = self.clip.std.Lut(0, function=lambda x: int(x / 2))

        preview = core.std.MaskedMerge(base, self.clip, self._zone_mask(self.coords))
        if self.coords_alt:
            preview = core.std.MaskedMerge(preview, self.clip, self._zone_mask(self.coords_alt))

        return preview


    @property
    def preview_crop(self) -> vs.VideoNode:
        """Preview of the cropped input"""

        if not self.coords_alt:
            return core.std.Crop(self.clip, *self.coords)
        else:
            top = core.std.Crop(self.clip, *self.coords_alt)
            bottom = core.std.Crop(self.clip, *self.coords)

            diff = tuple([int((top.width - bottom.width) / 2)] * 2)

            if top.width > bottom.width:
                bottom = bottom.std.AddBorders(*diff)
            elif top.width < bottom.width:
                top = top.std.AddBorders(*diff)

            return core.std.StackVertical([top, bottom])


    @property
    def preview_clean(self) -> vs.VideoNode:
        """Preview of the clean OCR output"""
        return self._cleaning(self.preview_crop)


    def _zone_mask(self, coords: Tuple[int, int, int, int]) -> vs.VideoNode:
        """Generates rectangular mask of the zone to OCR

        :param coords:      Amount of pixel to crop for each side.

        :return:            Mask of the zone.
        """
        left, right, top, bottom = coords

        return core.std.Crop(
            core.std.BlankClip(self.clip, color=self._scale_values(255, self.clip.format.bits_per_sample)),  # type: ignore[union-attr]
            *coords
        ).std.AddBorders(left, right, top, bottom)


    @staticmethod
    def _convert_coords(clip: vs.VideoNode, coords: Tuple[int, int, int], alt: bool = False) -> Tuple[int, int, int, int]:
        """Convert OCR coords to std.Crop coords

        :param coords:      coords of the zone to OCR

        :return:            std.Crop coords
        """

        width, height, offset = coords
        side_crop = int((clip.width - width) / 2)
        top_crop = int(clip.height - height - offset)
        bottom_crop = int(offset)

        if alt:
            top_crop, bottom_crop = bottom_crop, top_crop

        return (
            side_crop,
            side_crop,
            top_crop,
            bottom_crop
        )


    @staticmethod
    def _scale_values(n: int, target_bitdepth: int, src_bitdepth: int = 8) -> int:
        """Scale a value to the correct bitdepth. Doesn't work with float values.

        :param n:           value to scale
        :param bitdepth:    bitdepth to scale the value to

        :return:            scaled value
        """
        return n << (target_bitdepth - src_bitdepth)


    @staticmethod
    def _get_sub_headers(width: int, height: int) -> str:
        """Generate ASS headers with custom resolution.

        :param width:       Width of the source video.
        :param height:      Height of the source video.

        :returns:           ASS headers.
        """

        return f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Verdana,{int(height * 57 / 1080)},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,{round(height * 3.6 / 1080, 1)},0,2,100,100,{int(height * 80 / 1080)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
