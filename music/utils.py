import os
from music21 import converter, instrument, note, chord, stream
import pickle
import subprocess
import glob


def convertMidi2Mp3():
    # 将神经网络生成的MIDI文件转成MP3
    print("Converting to MP3")


def get_notes():
    #从misic_midi 目录中所有文件中读取note(音符： A, B A#, C#) 和 chord(和弦: [B4, E5 , G#4], [C5, E5], 多个note的集合)
    notes = []

    # glob:匹配所有符合条件的文件并且以 List 的形式返回
    for file in glob.glob("music_midi/*.mid"):
        # 读取MIDI，输出stream流类型
        stream = converter.parse(file)

        #获取所有乐器部分
        parts = instrument.partitionByInstrument(stream)

        if parts:
            #如果有乐器部分，取第一个乐器部分
            notes = parts.parts[0].recurse()
        else:
            notes = stream.flat.notes

        #打印出每一个元素的
        for element in notes:
            #如果是 note 类型,取它的音调（pitch)
            if isinstance(element, note.Note):
                #格式例如：E6
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # 格式为和弦，进行转换 -> 4.12.7
                notes.append('.'.join(str(n) for n in element.normalOrder))
            
    # 将数据写入data/notes文件
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def create_music(prediction):
    """
    用神经网络预测的音乐数据来生成 midi 文件，再转MP3
    """
    offset = 0
    output_notes = []

    #生成 Note 或 Chord
    for data in prediction:
        # 如果是 Chord 格式(3.15.3)
        if ('.' in data) or data.isdigit():
            notes_in_chord = data.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano() # 乐器使用钢琴
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        #每次迭代都将偏移增加，不会产生交叠覆盖
        offset = offset + 0.5

    # 创建音乐流
    midi_stream = stream.Stream(output_notes)

    #写入midi文件
    midi_stream.write('midi', fp='output.mid')

    # 将生成的midi 转换成MP3