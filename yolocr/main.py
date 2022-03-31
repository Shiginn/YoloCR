import argparse
import os
import warnings

import vapoursynth as vs

from .ocr import YoloCR

core = vs.core


def main():
    parser = argparse.ArgumentParser(description="OCR toolkit based on VapourSynth and Tesseract")

    parser.add_argument(
        "input_file", type=str, help="Path to your input file"
    )
    parser.add_argument(
        "-o", "--output", required=False, type=str, default="subs.ass",
        help="Path to your subtitle file (default: %(default)s)"
    )

    parser.add_argument(
        "-c", "--coords", required=False, nargs=3, type=int, default=[1500, 210, 50],
        help="Coordinates of the text (default: %(default)s)"
    )
    parser.add_argument(
        "-ca", "--coords_alt", required=False, nargs=3, type=int, default=None,
        help="Coordinates of the alt text (default: %(default)s)"
    )

    parser.add_argument(
        "-ti", "--thr_in", required=False, type=int, default=220,
        help="Binarization threshold of the subtitle inline (default: %(default)s)"
    )
    parser.add_argument(
        "-to", "--thr_out", required=False, type=int, default=70,
        help="Binarization threshold of the subtitle outline (default: %(default)s)"
    )

    parser.add_argument(
        "-l", "--lang", required=False, default="eng",
        help="Subtitle language"
    )

    parser.add_argument(
        "--indexer", required=False, choices=["ffms2", "lsmas", "auto"], default="auto",
        help="Input file indexer to use (default: %(default)s)"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"File {args.input_file} not found")

    if (file_format := os.path.splitext(args.input_file)[1]) not in [".mkv", ".mp4"]:
        warnings.warn("Input file format might not be supported")

    if args.indexer == "auto":
        args.indexer = "lsmas" if file_format == ".mp4" else "ffms2"

    if args.indexer == "lsmas":
        input_clip = core.lsmas.LWLibavSource(args.input_file)
    elif args.indexer == "ffms2":
        input_clip = core.ffms2.Source(args.input_file)

    ocr = YoloCR(
        input_clip,
        coords=args.coords,
        coords_alt=args.coords_alt,
        thr_in=args.thr_in,
        thr_out=args.thr_out,
    )

    ocr.extract_frames()
    ocr.write_subs(args.lang, args.output)


if __name__ == "__main__":
    main()
