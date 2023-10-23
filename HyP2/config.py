import argparse

def get_config():
    parser = argparse.ArgumentParser(description='retrieval')
    parser.add_argument('--dataset', type=str, default='nuswide',
                        help="dataset name")  # will be updated below
    parser.add_argument('--hash_bit', type=str, default='48', help="number of hash code bits")  # will be updated below
    parser.add_argument('--batch_size', type=int, default=100, help="batch size")
    parser.add_argument('--epochs', type=int, default=100, help="epochs")
    parser.add_argument('--cuda', type=int, default=0, help="cuda id")
    parser.add_argument('--backbone', type=str, default='alexnet', help="backbone")  # googlenet, resnet, alexnet
    parser.add_argument('--beta', type=float, default=0.5, help="hyper-parameter for regularization")
    parser.add_argument('--retrieve', type=int, default=0, help="retrieval number aka top-k")  # will be updated below
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--rate', type=float, default=0.02, help="lr rate")
    # add for tweaks
    parser.add_argument('--num_classes', type=int, default=10, help="num_classes")
    parser.add_argument('--feature_rate', type=float, default=0.01, help="feature_rate")
    parser.add_argument('--threshold', type=float, default=0.0, help="threshold")
    parser.add_argument('--save_dir', type=str, default=0.0, help="save_dir")

    return parser.parse_args()
