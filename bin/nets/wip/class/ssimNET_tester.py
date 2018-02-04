#!/usr/bin/env python3
import ssimNET

def main():
    test_net = ssimNET.ssimNET(data_path = '/home/dirty_mike/Dropbox/github/image_quality_analysis/data/sample_data/')
    test_net.load_data(local=True)



if __name__ == '__main__':
    main()
