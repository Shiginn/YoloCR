from vskernels import Point
from vstools import core, set_output

from yolocr import UnetCleaner, UnetModel, YoloCR

source_file = core.bs.VideoSource("/path/to/hardsubbed_video.ext")

yolocr = YoloCR(
    source_file,
    # replace UnetModel.SMALL with "path/to/your/model.onnx" to use a custom model
    cleaner=UnetCleaner(UnetModel.SMALL).with_postprocess(lambda clip: Point().supersample(clip, 2)),
    coords=(1280, 130, 0),
    coords_alt=False,
)


if __name__ == "__main__":
    # Run clean and subtitle detection
    yolocr.extract_frames()

    # Save results to disk
    yolocr.to_disk("filtered_images")

    # Generate PGS subtitle file
    yolocr.to_pgs("subs.sup")

else:
    set_output(source_file)
    set_output(yolocr.clip_coords)
    set_output(yolocr.clip_crop)
    set_output(yolocr.clip_clean)
