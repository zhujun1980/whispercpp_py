#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import logging
from ctypes import *
from platform import *


class LibraryNotFound(Exception):
    pass


def find_lib(base_path=None):
    cdll_names = {
        'Darwin': 'libwhisper.so',
        'Linux': 'libwhisper.so',
    }

    if base_path:
        lib_path = os.path.join(base_path, cdll_names[system()])
        if not os.path.exists(lib_path) or not os.path.isfile(lib_path):
            raise LibraryNotFound('Cannot find library in the path: ' + base_path)
        return lib_path

    file_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    dll_path = [
        os.getcwd(),
        file_dir,
        os.path.join(sys.prefix, 'lib'),
        os.path.join(sys.prefix, 'lib64')
    ]
    dll_path.extend(sys.path)
    dll_path = [os.path.join(p, cdll_names[system()]) for p in dll_path]
    lib_paths = [p for p in dll_path if os.path.exists(p) and os.path.isfile(p)]
    if not lib_paths:
        raise LibraryNotFound('Cannot find library in the path: \n' + ('\n'.join(dll_path)))
    return lib_paths[0]


class _whisper_context(Structure):
    pass


class _whisper_full_params_greedy(Structure):
    _fields_ = [("best_of", c_int)]


class _whisper_full_params_beam_search(Structure):
    _fields_ = [("beam_size", c_int),
                ("patience", c_float)]


class _whisper_full_params(Structure):
    _fields_ = [("strategy", c_int),
                ("n_threads", c_int),
                ("n_max_text_ctx", c_int),
                ("offset_ms", c_int),
                ("duration_ms", c_int),
                ("translate", c_bool),
                ("no_context", c_bool),
                ("single_segment", c_bool),
                ("print_special", c_bool),
                ("print_progress", c_bool),
                ("print_realtime", c_bool),
                ("print_timestamps", c_bool),
                ("token_timestamps", c_bool),
                ("thold_pt", c_float),
                ("thold_ptsum", c_float),
                ("max_len", c_int),
                ("split_on_word", c_bool),
                ("max_tokens", c_int),
                ("speed_up", c_bool),
                ("audio_ctx", c_int),
                ("initial_prompt", c_char_p),
                ("prompt_tokens", POINTER(c_int)),
                ("prompt_n_tokens", c_int),
                ("language", c_char_p),
                ("detect_language", c_bool),
                ("suppress_blank", c_bool),
                ("suppress_non_speech_tokens", c_bool),
                ("temperature", c_float),
                ("max_initial_ts", c_float),
                ("length_penalty", c_float),
                ("temperature_inc", c_float),
                ("entropy_thold", c_float),
                ("logprob_thold", c_float),
                ("no_speech_thold", c_float),
                ("greedy", _whisper_full_params_greedy),
                ("beam_search", _whisper_full_params_beam_search),

                ("new_segment_callback", c_void_p),
                ("new_segment_callback_user_data", c_void_p),

                ("progress_callback", c_void_p),
                ("progress_callback_user_data", c_void_p),

                ("encoder_begin_callback", c_void_p),
                ("encoder_begin_callback_user_data", c_void_p),

                ("logits_filter_callback", c_void_p),
                ("logits_filter_callback_user_data", c_void_p)]


class WhisperDLL:
    def __init__(self, base_path=None):
        lib_path = find_lib(base_path)
        logging.info("Library found:" + lib_path)
        self.dll = CDLL(lib_path)

        self.whisper_init_from_file = self.dll.whisper_init_from_file
        self.whisper_init_from_file.argtypes = [c_char_p]
        self.whisper_init_from_file.restype = POINTER(_whisper_context)

        self.whisper_free = self.dll.whisper_free
        self.whisper_free.argtypes = [POINTER(_whisper_context)]
        self.whisper_free.restype = None

        self.whisper_full_default_params = self.dll.whisper_full_default_params
        self.whisper_full_default_params.argtypes = [c_int]
        self.whisper_full_default_params.restype = _whisper_full_params

        self.whisper_full = self.dll.whisper_full
        self.whisper_full.argtypes = [POINTER(_whisper_context), _whisper_full_params, POINTER(c_float), c_int]
        self.whisper_full.restype = c_int

        self.whisper_full_parallel = self.dll.whisper_full_parallel
        self.whisper_full_parallel.argtypes = [POINTER(_whisper_context), _whisper_full_params, POINTER(c_float), c_int, c_int]
        self.whisper_full_parallel.restype = c_int

        self.whisper_full_n_segments = self.dll.whisper_full_n_segments
        self.whisper_full_n_segments.argtypes = [POINTER(_whisper_context)]
        self.whisper_full_n_segments.restype = c_int

        self.whisper_full_get_segment_text = self.dll.whisper_full_get_segment_text
        self.whisper_full_get_segment_text.argtypes = [POINTER(_whisper_context), c_int]
        self.whisper_full_get_segment_text.restype = c_char_p


whisper_dll = WhisperDLL(os.getenv("LIBWHISPER_BASE_PATH"))
