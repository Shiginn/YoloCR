import os
import re
import html
import pytesseract
import vapoursynth as vs
import xml.etree.ElementTree as ET
from functools import partial
from pytimeconv import Convert

from fractions import Fraction
from typing import Callable, Tuple, Union, Optional, List

core = vs.core

class OCR():

    def __init__(
        self,
        
        clip_hardsub: vs.VideoNode,
        
        coords: Tuple[int, int, int],
        coords_alt: Optional[Tuple[int, int, int]] = None,
        
        thr_in: Union[int, Tuple[int, int, int]] = 220,
        thr_out: Union[int, Tuple[int, int, int]] = 70,

        thr_sc: float = 0.0015,

        rect_size: int = 8
    ) -> None:
        
        """
        :param clip_hardsub:        Hardsubbed clip to OCR.
        
        :param coords:              Postion of the detection box (width, height, vertical margin from the bottom).
        
        :param coords_alt:          Postion of the alt detection box (width, height, vertical margin from the top). Will increase processing time.
        
        :param thr_in:              Binarization threshold of the subtitle text. Higher means less errors but text might not be detected. Should not be lower than subtitle text luminosity
        
        :param thr_out:             Binarization threshold of the subtitle outline. Higher means less errors will be removed but text might be detected as error. Should not be higher than subtitle outline luminosity.
        
        :param thr_sc:              Threshold the subtitle timing detection. Lower means more subtitles will be detected but might cause false positive.
        
        :param rect_size:           Size of the rectangle used to detect cleaning errors. Higher means more errors will be removed but might detect text as error.

        """

        self.clip = clip_hardsub

        thr_in = [thr_in] if isinstance(thr_in, int) else thr_in
        thr_out = [thr_out] if isinstance(thr_out, int) else thr_out


        if len(thr_in) != len(thr_out):
            raise ValueError("Both thr must have the same number of values")

        if len(thr_in) == 1:
            self.clip, *_ = core.std.SplitPlanes(clip_hardsub)
        
        self.thr_in = [self._scale_values(x, self.clip.format.bits_per_sample) for x in thr_in]
        self.thr_out = [self._scale_values(x, self.clip.format.bits_per_sample) for x in thr_out]

        self.thr_sc = thr_sc

        self.coords = self._convert_coords(self.clip, coords)
        self.coords_alt = self._convert_coords(self.clip, coords_alt, True) if coords_alt else None

        self.rect_size = rect_size


    def extract_frames(self):
        """Extract the subtitles in images"""

        self._write_frames(self._cleaning(self._crop(self.clip, self.coords)))

        if self.coords_alt:
            self._write_frames(self._cleaning(self._crop(self.clip, self.coords_alt)), alt=True)



    def write_subs(self, lang: str):
        """Write the ASS file from the extracted frames"""

        if lang not in pytesseract.get_languages():
            raise ValueError(f"Language '{lang}' is not installed")
        
        try:
            os.mkdir("tesseract_results")
        except FileExistsError:
            pass

        print("OCRing images...")

        file_to_process = os.listdir("filtered_images")
        from multiprocessing import Pool, cpu_count

        with open("subs.ass", "w") as sub_file:
            sub_file.write(self._get_sub_headers(self.clip.width, self.clip.height))

        with Pool(cpu_count()-1) as p:
            lines = p.map(
                partial(self._ocr_line, lang=lang), file_to_process
            )

        with open("subs.ass", "a", encoding="utf8") as sub_file:
            sub_file.write("\n".join(sorted(lines)))


    def _cleaning(self, clip: vs.VideoNode) -> vs.VideoNode:
        """Prepares the clip for the OCR by removing everything that is not subtitles
        
        :param clip:        clip to process
        """
        bnz_in = core.std.Binarize(clip, self.thr_in)
        bnz_out = core.std.Binarize(clip, self.thr_out)

        blank_clip = core.std.BlankClip(clip,
            width=clip.width - self.rect_size * 2,
            height=clip.height - self.rect_size * 2,
            color=[0] * clip.format.num_planes
        )

        rect = core.std.AddBorders(blank_clip,
            left=self.rect_size,
            right=self.rect_size,
            top=self.rect_size,
            bottom=self.rect_size,
            color = [self._scale_values(255, clip.format.bits_per_sample)] * clip.format.num_planes
        )

        overlap = core.std.Expr([rect, bnz_out], "x y min")

        ocr_issues = core.misc.Hysteresis(overlap, bnz_out).std.Invert()

        txt = core.std.MaskedMerge(
            core.std.BlankClip(clip),
            bnz_in,
            ocr_issues
        )

        if clip.format.num_planes == 3:
            txt = core.std.Expr(core.std.SplitPlanes(txt), "x y max z max")

        return txt.std.Minimum().std.Maximum()

    @staticmethod
    def _ocr_line(file: str, lang: str) -> str:
        """OCR a single image and returns the result in ASS format
        
        :param file:        path to the image to process
        :param lang:        language of the text to OCR

        :returns:           OCR'd text in ASS format
        """
        tesseract_hocr = pytesseract.image_to_pdf_or_hocr(f"filtered_images/{file}", lang, config="--oem 0 --psm 6", extension="hocr")

        hocr = html.unescape(tesseract_hocr.decode("utf8"))
        hocr = hocr.replace("<em>", "{\\i1}").replace("</em>", "{\\i0}")

        raw = "".join(
            list(ET.fromstring(hocr).itertext())
        ).replace("{\\i1} {\\i0}", "").strip()

        one_line = re.sub(r"\n{1} +", " ", raw)

        merge_italics = re.sub(
            r"{\\i0}\W+{\\i1}",
            lambda x: x.group()[5:-5],
            one_line
        )

        linebreak = re.sub(r" {2,}", r"\\N", merge_italics)

        corrections = [
            ("’", "'"), ("‘", "'"), (" '", "'"), ("!'", "l'"), ("I'", "l'"),
            ("_", "-"), ('—', '-'), ("...", "…"),
            ("<<", "«"), (">>", "»"), ("« ", "\""), (" »", "\""),
            ('II', 'Il'), ("iI", "il"),
        ]

        for correction in corrections:
            linebreak = linebreak.replace(*correction)


        if "_alt" in file:
            if linebreak.startswith("{"):
                linebreak = linebreak[:1] + r"\an8" + linebreak[1:]
            else:
                linebreak = r"{\an8}" + linebreak
        
        start_ts, end_ts, *_ = file.split("_")
        
        start_ts = start_ts.replace("-", ":")
        end_ts = end_ts.replace("-", ":")

        return f'Dialogue: 10,{start_ts},{end_ts},Default,,0,0,0,,{linebreak}'


    def _write_frames(self, clip: vs.VideoNode, alt: bool = False) -> None:
        """Write images with subtitles from processed clip
        
        :param clip:        processed clip to extract frames from
        :param alt:         use alt coords
        """
        self.results: List[int] = [0, clip.num_frames - 1]

        def _get_frame_ranges(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
            if f.props["_SceneChangePrev"] == 1 or f.props["_SceneChangeNext"] == 1:
                self.results.insert(-1, n)
            return clip

        ocr = core.std.FrameEval(
            clip,
            partial(_get_frame_ranges, clip=clip),
            prop_src=clip.misc.SCDetect(self.thr_sc)
        )

        print(f"Detecting subtitles in {'main' if not alt else 'alt'} clip")
        self._output_to_devnull(ocr, True, lambda x, y : print(f"{x}/{y}", end="\r"))


        self.results = sorted(self.results)
        self.results = [(self.results[i], self.results[i+1]) for i in range(0, len(self.results)-1, 2)][1::2]

        try:
            os.mkdir("filtered_images")
        except FileExistsError:
            pass

        print("\nWriting image files")
        clip = core.std.Invert(clip)

        for (start_f, end_f) in self.results:
            path = self._get_path(start_f, end_f+1, clip.fps_num/clip.fps_den, alt)

            if os.path.isfile(path):
                os.remove(path)

            frame = clip[start_f].imwri.Write(
                imgformat="PNG",
                filename=path#f"filtered_images/{start_f}_{end_f+1}{'_alt' if alt else ''}_%01d.png"
            )
            
            self._output_to_devnull(frame, y4m=False)


    @property
    def preview_coords(self) -> vs.VideoNode:
        """Preview of the OCR zone(s)"""
        base = self.clip.std.Lut(0, function=lambda x: int(x/2))

        preview = core.std.MaskedMerge(base, self.clip, self._zone_mask(self.coords))
        if self.coords_alt:
            preview = core.std.MaskedMerge(preview, self.clip, self._zone_mask(self.coords_alt))
        
        return preview

    
    @property
    def preview_crop(self) -> vs.VideoNode:
        """Preview of the cropped input"""

        if not self.coords_alt:
            return self._crop(self.clip, self.coords)
        else:
            top = self._crop(self.clip, self.coords_alt)
            bottom = self._crop(self.clip, self.coords)

            if top.width > bottom.width:
                bottom = bottom.std.AddBorders((top.width - bottom.width)/2, (top.width - bottom.width)/2)
            elif top.width < bottom.width:
                top = top.std.AddBorders((top.width - bottom.width)/2, (top.width - bottom.width)/2)

            return core.std.StackVertical([top, bottom])


    @property
    def preview_clean(self) -> vs.VideoNode:
        """Preview of the clean OCR output"""
        return self._cleaning(self.preview_crop)


    def _zone_mask(self, coords: Tuple[int, int, int, int]) -> vs.VideoNode:
        """Generates rectangular mask of the zone to OCR
        
        :param coords:      amount of pixel to crop for each side

        :return:            mask of the zone
        """
        
        left, right, top, bottom = coords
        
        return self._crop(
            core.std.BlankClip(self.clip, color=[self._scale_values(255, self.clip.format.bits_per_sample)] * self.clip.format.num_planes),
            coords
        ).std.AddBorders(left, right, top, bottom)


    @staticmethod
    def _crop(clip: vs.VideoNode, coords: Tuple[int, int, int, int]) -> vs.VideoNode:
        """Crop a clip to the input coords
        
        :param coords:      amount of pixel to crop for each side 

        :return:            croped clip
        """

        left, right, top, bottom = coords
        return clip.std.Crop(left, right, top, bottom)


    @staticmethod
    def _convert_coords(clip: vs.VideoNode, coords: Tuple[int, int, int], alt: bool = False) -> Tuple[int, int, int, int]:
        """Convert OCR coords to std.Crop coords
        
        :param coords:      coords of the zone to OCR

        :return:            std.Crop coords
        """

        width, height, offset = coords
        return (
            int((clip.width - width)/2),
            int((clip.width - width)/2),
            int(clip.height - height - offset) if not alt else int(offset),
            int(offset) if not alt else (clip.height - height - offset)
        )
    

    @staticmethod
    def _scale_values(n: int, bitdepth: int, src_bitdepth: int = 8) -> int:
        """Scale a value to the correct bitdepth
        
        :param n:           value to scale
        :param bitdepth:    bitdepth to scale the value to

        :return:            scaled value
        """
        return n << (bitdepth - src_bitdepth)


    @staticmethod
    def _get_path(start: int, end: int, fps: Fraction, alt: bool = False):
        start = Convert.f2assts(start, fps).replace(":", "-")
        end = Convert.f2assts(end, fps).replace(":", "-")

        return f"{os.getcwd()}/filtered_images/{start}_{end}{'_alt' if alt else ''}_%01d.png"
    

    @staticmethod
    def _output_to_devnull(clip: vs.VideoNode, y4m: bool, progress_update: Optional[Callable] = None):
        """Output a clip to devnull.
        
        :param clip:                Clip to output.
        :param y4m:                 Output in y4m.
        :clip progress_update:      Progress updage function.

        :returns:
        """
        with open(os.devnull, "wb") as devnull:
            clip.output(devnull, y4m, progress_update)
    

    @staticmethod
    def _get_sub_headers(width: int, height: int):
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