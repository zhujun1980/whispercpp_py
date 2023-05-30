#!/usr/bin/env python
# encoding: utf-8

from ctypes import *
from typing import Any
from whispercpp_py.dll import whisper_dll as _dll
import wave
import numpy as np

# enum whisper_sampling_strategy {
WHISPER_SAMPLING_GREEDY = 0
WHISPER_SAMPLING_BEAM_SEARCH = 1
# }


class InvalidWhisperStrategy(ValueError): ...


class WhisperParams:
    def __init__(self, strategy: int) -> None:
        if strategy not in [WHISPER_SAMPLING_GREEDY,
                            WHISPER_SAMPLING_BEAM_SEARCH]:
            raise InvalidWhisperStrategy("Unknown strategy value: {}".format(strategy))
        self.params = _dll.whisper_full_default_params(strategy)


class Whisper(object):
    def __init__(self, model_path) -> None:
        self.ctx = _dll.whisper_init_from_file(
                create_string_buffer(model_path.encode()))

    def __del__(self):
        _dll.whisper_free(self.ctx)

    def transcript_wave_file(self, wav_path: str, params: WhisperParams):
        with wave.open(wav_path) as wav_file:
            raw = wav_file.readframes(wav_file.getnframes())
            sample = np.frombuffer(raw, np.int16).flatten().astype(np.float32) / 32768.0
            return self.transcript_ndarray(sample, params)

    def transcript_ndarray(self, sample: np.ndarray, params: WhisperParams):
        ret = _dll.whisper_full(self.ctx, params.params, sample.ctypes.data_as(POINTER(c_float)), len(sample))
        return ret

    def get_n_segments_simple(self):
        n = _dll.whisper_full_n_segments(self.ctx)
        return [_dll.whisper_full_get_segment_text(self.ctx, i) for i in range(n)]
