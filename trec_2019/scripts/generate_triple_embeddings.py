import h5py
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('triple_path', type=str)
    parser.add_argument('query_emb_path', type=str)
    parser.add_argument('doc_emb_path', type=str)
    args = parser.parse_args()
    
    