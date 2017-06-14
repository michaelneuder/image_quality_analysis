#!/usr/bin/env python3
import numpy as np

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

with open('../../data/sample_data/binary_data/orig_500_bin.txt', mode='r') as read_file:
    bin_data = read_file.read()
read_file.close()

check = text_from_bits(bin_data)
for i in check:
    print(i)
# check = np.fromstring(check,dtype=np.uint8)
# check = np.reshape(check,(500,96,96))
# print(check.shape)
