from music21 import converter, instrument


def print_notes():
    # 读取MIDI，输出stream流类型
    stream = converter.parse('1.mid')

    #获取所有乐器部分
    parts = instrument.partitionByInstrument(stream)

    if parts:
        #如果有乐器部分，取第一个乐器部分
        notes = parts.parts[0].recurse()
    else:
        notes = stream.flat.notes

    #打印出每一个元素的
    for element in notes:
        print(element)

if __name__ == '__main__':
    print_notes()