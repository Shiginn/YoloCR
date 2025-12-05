import os
from functools import partial
from pathlib import Path
from shutil import rmtree

import numpy as np
from PIL import Image
from vstools import clip_async_render, core, scale_value, vs

from .cleaning.base import BaseCleaner
from .pgs import convert_frame_data, convert_images_data
from .types import CropCoords, ImageData, InputCoords

__all__ = ["YoloCR"]


class YoloCR:
    """OCR Class"""

    clip: vs.VideoNode
    cleaner: BaseCleaner

    coords: CropCoords
    coords_alt: CropCoords | None

    images: list[ImageData]

    def __init__(
        self,
        clip_hardsub: vs.VideoNode,
        cleaner: BaseCleaner,
        coords: InputCoords,
        coords_alt: InputCoords | bool = True,
    ) -> None:
        """
        :param clip_hardsub:        Hardsubbed clip to OCR.

        :param coords:              Postion of the bottom detection box
                                    (width, height, vertical margin from the bottom).

        :param coords_alt:          Postion of the top detection box (width, height, vertical margin from the top).
                                    Will increase processing time.
                                    If set to True, will use the same coords as `coords. If set to False, disables top
                                    subtitle detection. Defaults to True.
        """
        self.clip = clip_hardsub.resize.Bicubic(format=vs.YUV420P8)

        if self.clip.format is None:
            raise ValueError("Variable format clip are not supported.")

        if self.clip.format.color_family not in [vs.GRAY, vs.YUV]:
            raise ValueError("Input clip must be GRAY or YUV.")

        self.cleaner = cleaner

        self.coords = self._convert_coords(self.clip, coords)
        if coords_alt:
            coords_alt = coords
        self.coords_alt = self._convert_coords(self.clip, coords_alt, True) if coords_alt else None

        self.images = []

    def extract_frames(self) -> None:
        """
        Analyze the clip to find subtitles and the frames they appear in.

        :param write_to_disk:   Write each frame before OCR. Impact on performance is low (-5 fps) since
                                pytesseract writes temporary image if it doesn't already exist on the disk.
                                Defaults to False.
        """
        clean_clip = self.cleaner.run(core.std.Crop(self.clip, *self.coords))
        frame_ranges = self._extract_scene_frame_ranges(clean_clip.std.PlaneStats())
        self.images += self._write_sub_frames(clean_clip, frame_ranges)

        if self.coords_alt:
            clean_clip_alt = self.cleaner.run(core.std.Crop(self.clip, *self.coords_alt))
            frame_ranges_alt = self._extract_scene_frame_ranges(clean_clip_alt.std.PlaneStats(), alt=True)
            self.images += self._write_sub_frames(clean_clip_alt, frame_ranges_alt, alt=True)

    def to_disk(self, output_dir: str | Path) -> None:
        """Save extracted images to disk

        :param output_dir:  Directory to save images to.
        """
        try:
            os.mkdir("filtered_images")
        except FileExistsError:
            rmtree("filtered_images")
            os.mkdir("filtered_images")

        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        for image in self.images:
            image.data.save(output_dir / image.name)

    def to_pgs(self, output_file: str | Path) -> None:
        """Convert extracted images to PGS subtitle file.

        :param output_file:    Output PGS subtitle file.
        """
        if isinstance(output_file, str):
            output_file = Path(output_file)

        prepared_data = convert_images_data(self.images)
        size = (self.clip.width, self.clip.height)

        sub = convert_frame_data(prepared_data, size, (self.coords[2], self.coords_alt[2] if self.coords_alt else 0))

        with open(output_file, "wb") as f:
            f.write(sub)

    def _extract_scene_frame_ranges(self, clip: vs.VideoNode, alt: bool = False) -> list[tuple[int, int]]:
        """
        Get start and end frame of each subtitle line.

        :param clip:    Clip to process
        :param alt:     Whether or not to use alt clip. Defaults to False
        """

        scene_changes: list[tuple[int, int]] = []
        curr_start_scene: int | None = None
        # thr_sc = 0.0035 * (300000 / (clip.width * clip.height)) + self.thr_sc_offset

        def _get_frame_ranges(n: int, f: vs.VideoFrame, clip: vs.VideoNode) -> vs.VideoNode:
            nonlocal curr_start_scene
            nonlocal scene_changes

            scene_start = f.props["_SceneChangePrev"] == 1
            scene_end = f.props["_SceneChangeNext"] == 1

            has_text = f.props["PlaneStatsMax"] > 130 and f.props["PlaneStatsAverage"] > 0.0035

            # one frame subtitle is almost certainly false positive
            if scene_start and scene_changes == 1:
                return clip

            if (scene_start or n == 0) and has_text:
                curr_start_scene = n
            elif (scene_end) and has_text:
                if curr_start_scene is not None:
                    scene_changes.append((curr_start_scene, n))
                    curr_start_scene = None
            return clip

        ocr = core.std.FrameEval(clip, partial(_get_frame_ranges, clip=clip), prop_src=clip.misc.SCDetect(0.0035))
        clip_async_render(ocr, progress=f"Extracting {'bottom' if not alt else 'top'} subtitles...")

        return scene_changes

    def _write_sub_frames(
        self, clip: vs.VideoNode, frame_ranges: list[tuple[int, int]], alt: bool = False
    ) -> list[ImageData]:
        """Write images with subtitles from processed clip

        :param clip:        Cleaned clip to extract frames from.
        :param frames_nums: Frame ranges to extract. Must be a list of tuple (start_frame, end_frame)
        :param alt:         Whether or not to use alt coords. Defaults to False
        """
        images_data: list[ImageData] = []

        for start_f, end_f in frame_ranges:
            median = core.median.Median([clip[start_f], clip[int((start_f + end_f) // 2)], clip[end_f]])

            with median.get_frame(0) as f:
                f_array = np.asarray(f[0])
                img = Image.fromarray(f_array, mode="L")

            images_data.append(ImageData(start_f, end_f, alt, img))

        return images_data

    @property
    def clip_coords(self) -> vs.VideoNode:
        """Preview of the OCR zone(s)"""
        base = self.clip.std.Lut(0, function=lambda x: int(x / 2))

        preview = core.std.MaskedMerge(base, self.clip, self._zone_mask(self.coords))
        if self.coords_alt:
            preview = core.std.MaskedMerge(preview, self.clip, self._zone_mask(self.coords_alt))

        return preview

    @property
    def clip_crop(self) -> vs.VideoNode:
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
    def clip_clean(self) -> vs.VideoNode:
        """Preview of the clean OCR output"""
        return self.cleaner.run(self.clip_crop)

    def _zone_mask(self, coords: CropCoords) -> vs.VideoNode:
        """Generates rectangular mask of the zone to OCR

        :param coords:      Amount of pixel to crop for each side.

        :return:            Mask of the zone.
        """
        assert self.clip.format

        left, right, top, bottom = coords

        if self.clip.format.color_family == vs.GRAY:
            color = scale_value(255, 8, self.clip.format.bits_per_sample)
        else:
            color = [scale_value(255, 8, self.clip.format.bits_per_sample)] + [
                scale_value(128, 8, self.clip.format.bits_per_sample)
            ] * 2

        return core.std.Crop(
            core.std.BlankClip(self.clip, color=color),
            *coords,
        ).std.AddBorders(left, right, top, bottom)

    @staticmethod
    def _convert_coords(clip: vs.VideoNode, coords: InputCoords, alt: bool = False) -> CropCoords:
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

        return (side_crop, side_crop, top_crop, bottom_crop)
