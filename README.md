# whispercpp_py
Python binding for whisper.cpp

# Install

* Install whisper.cpp: [https://github.com/ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) first
* make libwhisper.so

# Run

```shell
LIBWHISPER_BASE_PATH=<Your libwhisper.so path> python test_lib.py
```

# Examples

## Wave

```Python
from ctypes import *
import whispercpp_py

model = "<Your model path>"
whisper = whispercpp_py.Whisper(model)

params = whispercpp_py.WhisperParams(whispercpp_py.WHISPER_SAMPLING_BEAM_SEARCH)
params.params.language = create_string_buffer(b"zh").value
params.params.print_special = False
params.params.print_progress = False
params.params.print_realtime = False
params.params.print_timestamps = False

wavfile = "<wave file>"
ret = whisper.transcript_wave_file(wavfile, params)
results = whisper.get_n_segments_simple()
[print(result.decode()) for result in results]
```

## OGG

```Python
from ctypes import *
import whispercpp_py
import base64
import numpy as np

import ffmpeg

model = "<Your model path>"
whisper = whispercpp_py.Whisper(model)

params = whispercpp_py.WhisperParams(whispercpp_py.WHISPER_SAMPLING_BEAM_SEARCH)
params.params.language = create_string_buffer(b"zh").value
params.params.print_special = False
params.params.print_progress = False
params.params.print_realtime = False
params.params.print_timestamps = False

oggfile = "<ogg file>"
mime = "audio/webm;codecs=opus"
with open(oggfile, "rb") as fp:
    content = fp.read()
    data_uri = "data:{};base64,{}".format(mime, base64.standard_b64encode(content).decode())
    _out, _ = (ffmpeg
                .input(data_uri, loglevel="0")
                .output("pipe:", format='s16le', acodec='pcm_s16le', ac=1, ar='16k', loglevel="0")
                .overwrite_output()
                .run(capture_stdout=True)
    )
    sample = (
        np
        .frombuffer(_out, np.int16)
        .flatten()
        .astype(np.float32)
        / 32768.0
    )
    status = whisper.transcript_ndarray(sample, params)
    results = whisper.get_n_segments_simple()
    [print(result.decode()) for result in results]


```
