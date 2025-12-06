from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from vsscale import autoselect_backend
from vstools import core, vs

from .base import BaseCleaner

if TYPE_CHECKING:
    from vsmlrt import backendT as Backend


class UnetModel(Enum):
    SMALL = "unet-32.onnx"
    MEDIUM = "unet-48.onnx"
    LARGE = "unet-64.onnx"


class UnetCleaner(BaseCleaner):
    model: Path
    backend: Backend

    def __init__(self, model: UnetModel | str | Path = UnetModel.SMALL, backend: Backend | None = None):
        """
        :param model:       Model to use for UNet cleaning. Defaults to a small pre-trained model. If a string or Path
                            is provided, it is treated as a path to a custom ONNX model.
        :param backend:     Backend to use for ONNX inference. Defaults to autoselecting the best backend for the
                            current system using `vsscale.autoselect_backend`.
        """
        super().__init__()

        if isinstance(model, UnetModel):
            model = Path(__file__).parent / "../models" / model.value
        elif isinstance(model, str):
            model = Path(model)

        self.model = model
        self.backend = backend if backend is not None else autoselect_backend(fp16=True)

    def _clean(self, clip: vs.VideoNode) -> vs.VideoNode:
        from vsmlrt import inference

        assert clip.format

        clip_float = clip.resize.Bicubic(format=vs.GRAYS)
        mask = inference(clip_float, str(self.model.resolve()), backend=self.backend)

        return core.std.Expr([clip_float, mask], "y 0.5 > x 0 ?").resize.Bicubic(format=vs.RGB24)
