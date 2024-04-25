import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--holoassist_dir', type=str, default="/Users/artemmerinov/data/holoassist")
parser.add_argument('--raw_annotation_file', type=str, default="/Users/artemmerinov/data/holoassist/data-annotation-trainval-v1_1.json")
parser.add_argument('--split_dir', type=str, default="/Users/artemmerinov/data/holoassist/data-splits-v1")
parser.add_argument('--fine_grained_actions_map_file', type=str, default="/Users/artemmerinov/data/holoassist/fine_grained_actions_map.txt")
parser.add_argument('--dataset_name', type=str, default="holoassist")
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--clip_gradient", type=float, default=None)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--memory_size", type=int, default=32)
parser.add_argument("--memory_dim", type=int, default=512)