import struct
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from AxxTerm_serial import BinaryStreamReader


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


if __name__ == '__main__':
    test_binary_reader_float32_little_endian()
    test_binary_reader_uint16_big_endian()
    test_binary_reader_partial_data()
    test_binary_reader_multiple_packages()
    test_binary_reader_sync_clears_buffer()
    print("All BinaryStreamReader tests passed!")
