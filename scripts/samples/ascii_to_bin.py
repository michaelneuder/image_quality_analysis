#!/usr/bin/env python3
import binascii
import numpy as np
np.set_printoptions(threshold=np.nan)

def main():
    with open('../../data/sample_data/orig_3pics.txt', mode ='r') as read_file:
        sample = read_file.read()
    read_file.close()

    bin_sample = bin(int.from_bytes(str(sample).encode(), 'big'))
    with open('../../data/sample_data/orig_3pics_bin.txt', mode='w') as write_file:
        write_file.write(bin_sample)
    write_file.close()

    with open('../../data/sample_data/orig_3pics_bin.txt', mode='r') as read_file:
        binary = read_file.read()
    read_file.close()
    back = np.fromstring(binary)
    print(back)



        # bin_sample = int(bin_sample, 2)
        # back = bin_sample.to_bytes((bin_sample.bit_length() + 7) // 8, 'big').decode()
        # print(back)

if __name__ == '__main__':
    main()
