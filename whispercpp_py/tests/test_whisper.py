#!/usr/bin/env python
# encoding: utf-8

import os
import unittest
from whispercpp_py.whispercpp import *

class TestWhisper(unittest.TestCase):
    def test_whisperparams(self):
        with self.assertRaises(InvalidWhisperStrategy):
            WhisperParams(2)

        params = WhisperParams(WHISPER_SAMPLING_BEAM_SEARCH)
        self.assertEqual(getattr(params.params, "strategy"), WHISPER_SAMPLING_BEAM_SEARCH)

        params = WhisperParams(WHISPER_SAMPLING_GREEDY)
        self.assertEqual(params.params.strategy, WHISPER_SAMPLING_GREEDY)

        params.params.language = create_string_buffer(b"zh").value
        self.assertEqual(params.params.language, b"zh")

    def test_whisper(self):
        model = "/Users/zhujun5/workspace/model/ggml-tiny.bin"
        filename = os.path.join(os.path.dirname(__file__), "a1.wav")
        params = WhisperParams(WHISPER_SAMPLING_BEAM_SEARCH)
        params.params.language = create_string_buffer(b"zh").value
        params.params.print_special = False
        params.params.print_progress = False
        params.params.print_realtime = False
        params.params.print_timestamps = False

        whisper = Whisper(model)
        ret = whisper.transcript_wave_file(filename, params)
        self.assertEqual(ret, 0)
        results = whisper.get_n_segments_simple()
        [print(result.decode()) for result in results]
