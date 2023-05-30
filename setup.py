#!/usr/bin/env python
# encoding: utf-8

import setuptools


__author__ = "Zhu Jun"
__version__ = "0.0.1"


setuptools.setup(
    name="whispercpp_py",
    author=__author__,
    version=__version__,
    author_email="zhujun1980@gmail.com",
    description="Python binding for whisper.cpp",
    long_description="Python binding for whisper.cpp <https://github.com/ggerganov/whisper.cpp>",
    url="https://github.com/zhujun1980/whispercpp_py",
    packages=setuptools.find_packages(),
    test_suite='whispercpp_py.tests',
)
