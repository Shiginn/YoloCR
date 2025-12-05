from dataclasses import dataclass

from PIL import Image

__all__ = ["InputCoords", "CropCoords"]

InputCoords = tuple[int, int, int]
CropCoords = tuple[int, int, int, int]


@dataclass
class ImageData:
    start: int
    end: int
    is_alt: bool
    data: Image.Image

    @property
    def name(self) -> str:
        return f"{self.start}_{self.end}{'_alt' if self.is_alt else ''}_ocr.png"
