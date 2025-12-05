from fractions import Fraction
from typing import Literal, TypeAlias, cast

from PIL import Image
from pytimeconv import Convert

from .types import ImageData

ColorValue: TypeAlias = Literal[0, 1, 2]

"""PGS subtitle generation from images. Taken from https://github.com/dam-cav/img-to-pgs-sup and modified."""

PG_ESCAPE = ("PG").encode("utf-8")
PG_SEGMENT_TYPE = {
    "PCS": 0x16,
    "WDS": 0x17,
    "PDS": 0x14,
    "ODS": 0x15,
    "END": 0x80,
}

# order is not casual, pay attention!
COLORS = [
    b"\x10\x80\x80\x00",  # 0 - transparent
    b"\xff\x7f\x7f\xff",  # 1 - white
    b"\x00\x7f\x7f\xff",  # 2 - black
]


def paletteCodeNumber(pixel: tuple[int, int]) -> ColorValue:
    if pixel[1] == 0:
        return 0  # transparent
    if pixel[0] == 0:
        return 0  # black converted to transparent
    # if pixel[0] == 0:
    #     return 2  # black
    return 1  # white


def toColorCode(colorcode: ColorValue, count: int) -> bytearray:
    # return noting on no repetitions
    if count == 0:
        return bytearray()

    # single pixel on non-base color (non-zero)
    if count == 1 and colorcode != 0:
        return bytearray(colorcode.to_bytes(1, "big"))

    structure = bytearray([0x00])  # default escape character
    if count <= 63:
        # 0     - 00 LL LL LL
        # 128   - 10 LL LL LL
        structure += ((count + 0) if colorcode == 0 else (count + 128)).to_bytes(1, "big")
    elif count <= 16383:
        # 16384 - 01 LL LL LL  LL LL LL LL
        # 49152 - 11 LL LL LL  LL LL LL LL
        structure += ((count + 16384) if colorcode == 0 else (count + 49152)).to_bytes(2, "big")
    else:
        # generate one of max size
        # missing 'count' will be passed to next block
        structure = toColorCode(colorcode, 16383)

    # need to specify color in case of non-base
    if colorcode != 0:
        # structure += numberToHexBytes(colorcode, 1)
        structure += colorcode.to_bytes(1, "big")

    if count > 16383:
        # recursively pass missing 'count' to next block
        structure += toColorCode(colorcode, count - 16383)

    return structure


def to_sub_data(image: Image.Image, size: tuple[int, int]) -> tuple[bytes, int, int]:
    data = bytearray()
    lastcolor: Literal[0, 1, 2] | None = None

    # reduce sub image to screen limit
    image.thumbnail(size)
    # reduce number of colors (LA = scale of grey with alpha-channel)
    image = image.convert("LA", dither=None)

    wid = image.width
    hei = image.height

    # same color pixel counter
    count = 0
    # current horizontal line pixel counter
    linewidth_count = 0

    # count pixel form left to right, top to bottom
    for pixel in iter(image.getdata()):
        pixel = cast(tuple[int, int], pixel)
        cycleColor = paletteCodeNumber(pixel)

        if lastcolor == cycleColor and linewidth_count < wid:
            # same color as before and horizontal line no ended
            count += 1
        else:
            # different color or line ended

            if lastcolor is not None:
                # write counted pixels (skipped on first cicle)
                data += toColorCode(lastcolor, count)

            # reset counters for next sum
            count = 1
            lastcolor = cycleColor

        linewidth_count += 1

        if linewidth_count > wid:
            # line end delimiter
            data.extend([0x00, 0x00])
            linewidth_count = 1

    # write last cycle color (since you will not found a different one next)
    assert lastcolor is not None
    data += toColorCode(lastcolor, count)

    return (data, wid, hei)


def generate_header(start_time: int, segment_type: int, segment_size: int) -> bytearray:
    header = bytearray()
    header += PG_ESCAPE  # Magic Number
    # 4 start time = seconds * 1000 (ms) * 90 (hz)
    header += start_time.to_bytes(4, "big")
    # 4 decoding time, always 0
    header.extend([0x00, 0x00, 0x00, 0x00])
    # 1 segment type
    header.append(segment_type)
    # 2 segment size
    header += segment_size.to_bytes(2, "big")
    return header


def generate_pcs(
    time: int,
    size: tuple[int, int],
    offset: int,
    ods_width: int,
    ods_height: int,
    counter: int,
    composition_type: Literal[0x00, 0x80],
) -> bytearray:
    pcs = generate_header(time, PG_SEGMENT_TYPE["PCS"], 19 if composition_type == 0x80 else 11)

    # PCS - CONTENT
    # 2 video width
    pcs += size[0].to_bytes(2, "big")
    # 2 video height
    pcs += size[1].to_bytes(2, "big")
    # 1 framerate
    pcs.append(0x10)
    # 2 composition number
    pcs += counter.to_bytes(2, "big")

    # 1 types of composition
    # pcs += b'\x00' # normal - delete previous block
    # pcs += b'\x40' # acquisition point - refresh
    # pcs += b'\x80' # epoch start - new block
    pcs.append(composition_type)

    # 1 palette update flag
    pcs.append(0x00)  # False
    # pcs += b'\x80' # True

    # 1 palette ID
    # 1 composition object number
    if composition_type == 0x00:
        pcs.extend([0x00, 0x00])
        return pcs
    else:
        pcs.extend([0x00, 0x01])

    # 2 object ID
    # 1 window ID
    pcs.extend([0x00, 0x00, 0x00])

    # 1 cropped
    # pcs.append(0x40)  # cropped
    pcs.append(0x00)  # non-cropped

    # print(size, ods_width, ods_height, offset)

    # 2 X offset in pixel
    pcs += round(size[0] / 2 - ods_width / 2).to_bytes(2, "big")
    # 2 Y offset in pixel
    # y_offset, is_alt = offset
    # pgs_offset = size[1] - ods_height - y_offset if not is_alt else y_offset
    pcs += offset.to_bytes(2, "big")
    # pcs += round(size[1] - ods_height - 10).to_bytes(2, "big")

    return pcs


def generate_wds(time: int, size: tuple[int, int], ods_width: int, ods_height: int) -> bytearray:
    wds = generate_header(time, PG_SEGMENT_TYPE["WDS"], 10)

    # WDS - CONTENT
    # 1 number of windows defined
    wds.append(0x01)
    # 1 window ID
    wds.append(0x00)
    # NOTE same as PCS
    # 2 window x offset (hor)
    wds += round(size[0] / 2 - ods_width / 2).to_bytes(2, "big")
    # 2 window y offset (vert)
    wds += round(size[1] - ods_height - 10).to_bytes(2, "big")
    # 2 width
    wds += ods_width.to_bytes(2, "big")
    # 2 height
    wds += ods_height.to_bytes(2, "big")

    return wds


def generate_pds(time: int, colors: list[bytes]) -> bytearray:
    pds = generate_header(time, PG_SEGMENT_TYPE["PDS"], 2 + 5 * len(colors))

    # PDS - CONTENT
    # 1 palette ID
    pds.append(0x00)
    # 1 palette version number
    pds.append(0x00)

    # describe each color in palette
    for i, color in enumerate(colors):
        pds += i.to_bytes(1, "big") + color

    return pds


def generate_ods(time: int, data: bytes, ods_width: int, ods_height: int) -> bytearray:
    ods = generate_header(time, PG_SEGMENT_TYPE["ODS"], 11 + len(data))

    # ODS - CONTENT
    # 2 object ID
    ods.extend([0x00, 0x00])
    # 1 object version number
    ods.append(0x00)

    # 1 sequence
    # ods += b'\x40' # last in sequence
    # ods += b'\x80' # first in sequence
    ods.append(0xC0)  # first and last in sequence

    # 3 data lenght
    # NOTE +4, discovered by studying real PGS
    ods += (len(data) + 4).to_bytes(3, "big")

    # 2 width of image
    ods += ods_width.to_bytes(2, "big")
    # 2 height of image
    ods += ods_height.to_bytes(2, "big")

    # data
    ods += data

    return ods


def generate_frame(
    image: Image.Image, start_time: int, end_time: int, counter: int, size: tuple[int, int], offset: int
) -> bytes:
    data, ods_width, ods_height = to_sub_data(image, size)

    pcs = generate_pcs(start_time, size, offset, ods_width, ods_height, counter, 0x80)
    wds = generate_wds(start_time, size, ods_width, ods_height)
    pds = generate_pds(start_time, COLORS)
    ods = generate_ods(start_time, data, ods_width, ods_height)
    end = generate_header(start_time, PG_SEGMENT_TYPE["END"], 0)

    end_pcs = generate_pcs(end_time, size, offset, ods_width, ods_height, counter + 1, 0x00)
    end_wds = generate_wds(end_time, size, ods_width, ods_height)
    end_end = generate_header(0, PG_SEGMENT_TYPE["END"], 0)

    sub = pcs + wds + pds + ods + end + end_pcs + end_wds + end_end
    return sub


def convert_images_data(images: list[ImageData], bin_thr: int = 128) -> list[ImageData]:
    prepared_files: list[ImageData] = []

    for image_data in images:
        ts = Convert.f2ts(image_data.start, Fraction(24000, 1001)).replace(".", ":").split(":")
        te = Convert.f2ts(image_data.end, Fraction(24000, 1001)).replace(".", ":").split(":")

        start_time = (
            (int(ts[0]) * 60 * 60 * 1000 * 90)
            + (int(ts[1]) * 60 * 1000 * 90)
            + (int(ts[2]) * 1000 * 90)
            + (int(ts[3]) * 90)
        )
        end_time = (
            (int(te[0]) * 60 * 60 * 1000 * 90)
            + (int(te[1]) * 60 * 1000 * 90)
            + (int(te[2]) * 1000 * 90)
            + (int(te[3]) * 90)
        )

        prepared_files.append(
            ImageData(
                start_time,
                end_time,
                image_data.is_alt,
                image_data.data.point(lambda x: 255 if x > bin_thr else 0),
            )
        )

    prepared_files.sort(key=lambda x: x.start)
    return prepared_files


def convert_frame_data(
    prepared_data: list[ImageData], frame_size: tuple[int, int], sub_offsets: tuple[int, int]
) -> bytes:
    sub = bytearray()

    for i, data in enumerate(prepared_data):
        # start_time, end_time, image = data
        # key = f"{start_time}_{end_time}"
        # print(f"Doing frame: {data.name}")

        # image = Image.open(file)
        # image = image.convert("RGB")

        sub += generate_frame(
            data.data,
            data.start,
            data.end,
            i,
            frame_size,
            sub_offsets[0] if not data.is_alt else sub_offsets[1],
        )

    return sub


# def main():
#     # print(dir_list)
#     # dir_list.sort()

#     counter = 0
#     sub = b""
#     pgs_frames: list[tuple[int, int, str]] = []

#     for file in dir_list:
#         time_match = re.search(
#             "(\\d{0,5})_(\\d{0,5})_ocr.png",
#             file,
#         )

#         start_frame = int(time_match.group(1))
#         end_frame = int(time_match.group(2))

#         ts = Convert.f2ts(start_frame, Fraction(24000, 1001))
#         te = Convert.f2ts(end_frame, Fraction(24000, 1001))

#         ts = ts.replace(".", ":").split(":")
#         te = te.replace(".", ":").split(":")

#         start_time = (
#             (int(ts[0]) * 60 * 60 * 1000 * 90)
#             + (int(ts[1]) * 60 * 1000 * 90)
#             + (int(ts[2]) * 1000 * 90)
#             + (int(ts[3]) * 90)
#         )
#         end_time = (
#             (int(te[0]) * 60 * 60 * 1000 * 90)
#             + (int(te[1]) * 60 * 1000 * 90)
#             + (int(te[2]) * 1000 * 90)
#             + (int(te[3]) * 90)
#         )

#         pgs_frames.append((start_time, end_time, file))
#         # if not pgs_frames.get(key):
#         #     pgs_frames[key] = []

#     pgs_frames.sort(key=lambda x: x[0])

#     for group_key in pgs_frames:
#         start_time, end_time, file = group_key
#         key = f"{start_time}_{end_time}"
#         print(f"Doing frame: {key} ({file})")
#         # print(" - images:", ", ".join(pgs_frames[group_key]))
#         # print(" - images:", len(group))

#         # image = mergeImageGroupVertically(group, path=ARGS.path, limiter=LIMITER)
#         image = Image.open(ARGS.path + LIMITER + file)
#         image = image.convert("RGB")
#         # image = addBorderToImage(image)

#         # time_match = re.search("([0-9]+)_([0-9]+)", group_key)

#         # start_time = int(time_match.group(1))
#         # end_time = int(time_match.group(2))

#         sub += generate_frame(file, image, start_time, end_time, counter, sizelimit)
#         counter += 1

#     with open(ARGS.outfile, "wb") as f:
#         f.write(sub)


# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     main()
