import struct
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from AxxTerm_serial import BinaryStreamReader, FrameReader


def test_binary_reader_float32_little_endian():
    reader = BinaryStreamReader()
    reader.data_type = 'float32'
    reader.endianness = 'little'
    reader.num_channels = 2
    data = struct.pack('<ff', 1.5, -3.0)
    results = reader.feed(data)
    assert len(results) == 1
    assert abs(results[0][0] - 1.5) < 1e-6
    assert abs(results[0][1] - (-3.0)) < 1e-6


def test_binary_reader_uint16_big_endian():
    reader = BinaryStreamReader()
    reader.data_type = 'uint16'
    reader.endianness = 'big'
    reader.num_channels = 3
    data = struct.pack('>HHH', 100, 200, 300)
    results = reader.feed(data)
    assert len(results) == 1
    assert results[0] == (100, 200, 300)


def test_binary_reader_partial_data():
    reader = BinaryStreamReader()
    reader.data_type = 'int16'
    reader.endianness = 'little'
    reader.num_channels = 2
    full = struct.pack('<hh', 42, -42)
    results = reader.feed(full[:3])
    assert len(results) == 0
    assert len(reader.buffer) == 3
    results = reader.feed(full[3:])
    assert len(results) == 1
    assert results[0] == (42, -42)


def test_binary_reader_multiple_packages():
    reader = BinaryStreamReader()
    reader.data_type = 'uint8'
    reader.endianness = 'little'
    reader.num_channels = 2
    data = bytes([10, 20, 30, 40, 50, 60])
    results = reader.feed(data)
    assert len(results) == 3
    assert results[0] == (10, 20)
    assert results[1] == (30, 40)
    assert results[2] == (50, 60)


def test_binary_reader_sync_clears_buffer():
    reader = BinaryStreamReader()
    reader.data_type = 'float32'
    reader.endianness = 'little'
    reader.num_channels = 1
    reader.feed(b'\x01\x02')
    assert len(reader.buffer) == 2
    reader.sync()
    assert len(reader.buffer) == 0


def test_frame_reader_fixed_size():
    reader = FrameReader()
    reader.data_type = 'uint16'
    reader.endianness = 'little'
    reader.num_channels = 2
    reader.sync_word = bytes([0xAA])
    reader.size_field = 'fixed'
    reader.frame_size = 4
    reader.checksum_enabled = False
    reader.reset()
    payload = struct.pack('<HH', 1000, 2000)
    data = bytes([0xAA]) + payload
    results = reader.feed(data)
    assert len(results) == 1
    assert results[0] == (1000, 2000)


def test_frame_reader_1byte_size():
    reader = FrameReader()
    reader.data_type = 'uint8'
    reader.endianness = 'little'
    reader.num_channels = 3
    reader.sync_word = bytes([0xAA, 0xBB])
    reader.size_field = '1-byte'
    reader.checksum_enabled = False
    reader.reset()
    payload = bytes([10, 20, 30])
    data = bytes([0xAA, 0xBB, 3]) + payload
    results = reader.feed(data)
    assert len(results) == 1
    assert results[0] == (10, 20, 30)


def test_frame_reader_2byte_size_big_endian():
    reader = FrameReader()
    reader.data_type = 'int16'
    reader.endianness = 'big'
    reader.num_channels = 2
    reader.sync_word = bytes([0xFF])
    reader.size_field = '2-byte'
    reader.checksum_enabled = False
    reader.reset()
    payload = struct.pack('>hh', -100, 200)
    size_bytes = struct.pack('>H', 4)
    data = bytes([0xFF]) + size_bytes + payload
    results = reader.feed(data)
    assert len(results) == 1
    assert results[0] == (-100, 200)


def test_frame_reader_checksum_pass():
    reader = FrameReader()
    reader.data_type = 'uint8'
    reader.endianness = 'little'
    reader.num_channels = 2
    reader.sync_word = bytes([0xAA])
    reader.size_field = 'fixed'
    reader.frame_size = 2
    reader.checksum_enabled = True
    reader.reset()
    payload = bytes([10, 20])
    checksum = sum(payload) & 0xFF
    data = bytes([0xAA]) + payload + bytes([checksum])
    results = reader.feed(data)
    assert len(results) == 1
    assert results[0] == (10, 20)


def test_frame_reader_checksum_fail():
    reader = FrameReader()
    reader.data_type = 'uint8'
    reader.endianness = 'little'
    reader.num_channels = 2
    reader.sync_word = bytes([0xAA])
    reader.size_field = 'fixed'
    reader.frame_size = 2
    reader.checksum_enabled = True
    reader.reset()
    payload = bytes([10, 20])
    data = bytes([0xAA]) + payload + bytes([0xFF])
    results = reader.feed(data)
    assert len(results) == 0


def test_frame_reader_multi_byte_sync_word():
    reader = FrameReader()
    reader.data_type = 'float32'
    reader.endianness = 'little'
    reader.num_channels = 1
    reader.sync_word = bytes([0xAA, 0xBB, 0xCC])
    reader.size_field = 'fixed'
    reader.frame_size = 4
    reader.checksum_enabled = False
    reader.reset()
    payload = struct.pack('<f', 3.14)
    data = bytes([0x00, 0x01, 0x02]) + bytes([0xAA, 0xBB, 0xCC]) + payload
    results = reader.feed(data)
    assert len(results) == 1
    assert abs(results[0][0] - 3.14) < 1e-5


def test_frame_reader_invalid_size_discards():
    reader = FrameReader()
    reader.data_type = 'uint16'
    reader.endianness = 'little'
    reader.num_channels = 2
    reader.sync_word = bytes([0xAA])
    reader.size_field = '1-byte'
    reader.checksum_enabled = False
    reader.reset()
    data = bytes([0xAA, 3, 0, 0, 0])
    results = reader.feed(data)
    assert len(results) == 0


def test_frame_reader_consecutive_frames():
    reader = FrameReader()
    reader.data_type = 'uint8'
    reader.endianness = 'little'
    reader.num_channels = 2
    reader.sync_word = bytes([0xAA])
    reader.size_field = 'fixed'
    reader.frame_size = 2
    reader.checksum_enabled = False
    reader.reset()
    data = bytes([0xAA, 10, 20, 0xAA, 30, 40])
    results = reader.feed(data)
    assert len(results) == 2
    assert results[0] == (10, 20)
    assert results[1] == (30, 40)


if __name__ == '__main__':
    test_binary_reader_float32_little_endian()
    test_binary_reader_uint16_big_endian()
    test_binary_reader_partial_data()
    test_binary_reader_multiple_packages()
    test_binary_reader_sync_clears_buffer()
    print("All BinaryStreamReader tests passed!")

    test_frame_reader_fixed_size()
    test_frame_reader_1byte_size()
    test_frame_reader_2byte_size_big_endian()
    test_frame_reader_checksum_pass()
    test_frame_reader_checksum_fail()
    test_frame_reader_multi_byte_sync_word()
    test_frame_reader_invalid_size_discards()
    test_frame_reader_consecutive_frames()
    print("All FrameReader tests passed!")
