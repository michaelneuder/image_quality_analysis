#!/usr/bin/env python3
import binascii
import numpy as np
np.set_printoptions(threshold=np.nan)

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

def main():
    with open('../../data/sample_data/orig_500.txt', mode ='r') as read_file:
        ascii_data = read_file.read()
    read_file.close()

    bin_data = text_to_bits(ascii_data)
    with open('../../data/sample_data/binary_data/orig_500_bin.txt', mode='w') as write_file:
        write_file.write(bin_data)
    write_file.close()


if __name__ == '__main__':
    main()
