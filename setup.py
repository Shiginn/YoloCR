#!/usr/bin/env python3

from setuptools import setup

with open("requirements.txt", "r", encoding="UTF-8") as r_file:
    req = r_file.read()

NAME = "yolocr"
VERSION = "0.0.1"

setup(
    name=NAME,
    version=VERSION,
    author="Shigin",
    author_email="shigin.contact@gmail.com",
    description="OCR toolkit based on VapourSynth and Tesseract",

    packages=["yolocr"],
    package_data={
        "yolocr": ['py.typed'],
    },

    install_requires=req,
    python_requires='>=3.9',

    entry_points={
        'console_scripts': [
            'yolocr = yolocr.main:main'
        ]
    }
)
