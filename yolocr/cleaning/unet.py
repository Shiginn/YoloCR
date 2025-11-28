from enum import Enum
from pathlib import Path

from vsmlrt import BackendV2, inference
from vstools import core, vs

from .abstract import AbstractCleaner


class UnetModel(Enum):
    SMALL = "unet-32.onnx"


class UnetCleaner(AbstractCleaner):
    model: UnetModel
    backend: BackendV2

    def __init__(self, model: UnetModel = UnetModel.SMALL, backend: BackendV2 | None = None):
        """
        :param model:  Model to use for UNet cleaning. Defaults to a small pre-trained model.
        :param backend:   Backend to use for ONNX inference. Defaults to TensorRT with FP16.
        """
        super().__init__()

        self.model = model
        self.backend = backend if backend is not None else BackendV2.TRT(fp16=True)

    def clean(self, clip: vs.VideoNode):
        assert clip.format

        model_path = Path(__file__).parent / "../models" / self.model.value
        clip_float = clip.resize.Bicubic(format=vs.GRAYS)
        mask = inference(clip_float, model_path.resolve(), backend=self.backend)

        return core.std.Expr([clip_float, mask], "y 0.5 > x 0 ?").resize.Bicubic(format=vs.RGB24)
