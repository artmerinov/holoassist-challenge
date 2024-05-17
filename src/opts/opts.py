import argparse


parser = argparse.ArgumentParser()

# ========================= Paths ==========================
parser.add_argument("--holoassist_dir", type=str, default="/Users/artemmerinov/data/holoassist", help="Path to dataset directory.")
parser.add_argument("--raw_annotation_file", type=str, default="/Users/artemmerinov/data/holoassist/data-annotation-trainval-v1_1.json", help="Path to raw annotation file.")
parser.add_argument("--split_dir", type=str, default="/Users/artemmerinov/data/holoassist/data-splits-v1", help="Path to split directory.")
parser.add_argument("--fga_map_file", type=str, default="/Users/artemmerinov/data/holoassist/fine_grained_actions_map.txt", help="Path to fine-grained action map file.")

# ========================= Model ==========================
parser.add_argument("--base_model", type=str, default="InceptionV3", choices=["ResNet50", "ResNet101", "InceptionV3", "TimeSformer", "HORST"])
parser.add_argument("--fusion_mode", type=str, default=None, choices=["None", "GSF", "GSM"], help="Fusion mode.")
parser.add_argument("--num_segments", type=int, default=8, help="Number of sampled frames from each video.")
parser.add_argument("--num_classes", type=int, default=1887, choices=[1887, 2], help="Number of labels.")

# ========================= Load Model =========================
parser.add_argument("--resume", type=str, default=None, help="Path to model to continue learning.")
parser.add_argument('--start-epoch', default=0, type=int, help="Epoch number.")

# ========================= Learning =========================
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay.")
parser.add_argument("--clip_gradient", type=float, default=None, help="Gradient norm clipping.")

# ========================= Dataloader ==========================
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers in dataloader.")
parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches loaded in advance by each worker.")
parser.add_argument("--debug", type=bool, default=False, help="Dubug mode, where only small portion of data is used.")
