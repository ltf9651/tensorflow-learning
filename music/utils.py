"""
MIDI相关函数
"""

import os
from midi2audio import FluidSynth


def convertMidi2Mp3():
    # 将神经网络生成的MIDI文件转成MP3
    input_file = "output.mid"
    output_file = "output.wav"
    fs = FluidSynth()
    fs.midi_to_audio(input_file, output_file)
    assert os.path.exists(input_file)

    print("Converting %s to MP3" % input_file)


if __name__ == '__main__':
    convertMidi2Mp3()
