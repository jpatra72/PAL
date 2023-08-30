import argparse
import gensim
import fasttext
import fasttext.util
from pathlib import Path, PureWindowsPath




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='de', help="shorthand of the language who's embedding is requrired")
    parser.add_argument('--dim_new', type=int, default=20, help="dimension of the reduced embedding")
    # parser.add_argument('--output_flname', type=str, default=)
    return parser.parse_args()


def main():
    args = parse_args()
    new_dim = args.dim_new
    emb_bin_filename = f"./data/emb/cc.{args.lang}.300.bin"

    emb_bin_filename_new = f"./data/emb/cc.{args.lang}.{new_dim}.bin"
    emb_vec_filename_new = f"./data/emb/cc.{args.lang}.{new_dim}.vec"

    # fasttext.util.download_model(args.lang, if_exists='ignore')
    ft = fasttext.load_model(emb_bin_filename)
    ft = fasttext.util.reduce_model(ft, new_dim)

    ft.save_model(emb_bin_filename_new)

    ft_gs = gensim.models.fasttext.load_facebook_model(emb_bin_filename_new)
    ft_gs.wv.save_word2vec_format(emb_vec_filename_new)

    pass

if __name__ == '__main__':
    main()
