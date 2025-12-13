# pOCR

pOCR is an OCR pre-processor to extract subtitles from hardsubbed anime. The goal of the project is not to provide an end-to-end OCR toolkit but provide the best source possible to your favorite OCR tool like Subtitle Extractor, Paddle OCR, your favorite LLM...

## Installation

To use pOCR, you need to install [VapourSynth](https://www.vapoursynth.com).

Install the required plugins :

```sh
vsrepo.py install misc median
```

> [!NOTE]
> If you want to use the Unet cleaner, you will also need to install [vs-mlrt](https://github.com/AmusementClub/vs-mlrt).

Finally, install pOCR :

```sh
pip install git+https://github.com/Shiginn/YoloCR.git@v2-refactor
```

## Usage

See [examples](examples/) directory.

## Cleaners

pOCR currently has 2 cleaners :

| Cleaner | Speed                                                           | Accuracy                                                                               |
| ------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| YoloCR  | Fast                                                            | Ranges from quite bad to ok/good. More likely to have false positive.                  |
| Unet    | From very fast to quite slow depending on the size of the model | Very good on situations the model has encountered during training. Very bad otherwise. |

The Unet models should be better and faster if the style of your subtitle is standard (no weird font, no extra slim font, no inverted colors) and your source is 720p or higher (upscaling can help the model produce better results).
If Unet struggles to pick-up the subtitles, try YoloCR. If both fail, good luck.

## Exporting

You can export the extracted subtitles in 2 formats :

- PNG images files
- PGS/SUP subtitle file

> [!WARNING]
> PGS support is very rudimentary. It only support binary images (binarization done internally) and pure black will be converted to transparent.
