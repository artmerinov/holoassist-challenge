import argparse


parser = argparse.ArgumentParser()

# ========================= Paths ==========================
parser.add_argument("--holoassist_dir", type=str, default="/Users/artemmerinov/data/holoassist", help="Path to dataset directory.")
parser.add_argument("--raw_annotation_file", type=str, default="/Users/artemmerinov/data/holoassist/data-annotation-trainval-v1_1.json", help="Path to raw annotation file.")
parser.add_argument("--split_dir", type=str, default="/Users/artemmerinov/data/holoassist/data-splits-v1", help="Path to split directory.")
parser.add_argument("--fine_grained_actions_map_file", type=str, default="/Users/artemmerinov/data/holoassist/fine_grained_actions_map.txt", help="Path fine-grained actions map file.")

# ========================= Dataset ==========================
parser.add_argument("--dataset_name", type=str, default="holoassist", help="Dataset name.")
parser.add_argument("--fusion_mode", type=str, default=None, choices=["None", "GSF", "GSM"], help="Fusion mode.")

# ========================= Model ==========================
parser.add_argument("--base_model", type=str, default="BNInception", choices=["resnet50", "resnet101", "BNInception", "InceptionV3", "TimeSformer"])
parser.add_argument("--num_segments", type=int, default=8, help="Number of sampled frames from each video.")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout ratio of last fc layer of base model.")

# ========================== Load Model ==========================
parser.add_argument("--resume", type=str, default=None, help="Path to model to continue learning.")
parser.add_argument('--start-epoch', default=0, type=int, help="Epoch number.")

# ========================= Learning ==========================
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum.")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay.")
parser.add_argument("--clip_gradient", type=float, default=None, help="Gradient norm clipping.")

# ========================= Monitor ==========================
parser.add_argument("--checkpoint_interval", type=int, default=3, help="How often the model is saved.")
parser.add_argument("--runs_path", type=str, default="runs/", help="Path to tensorboard runs directory.")

# ========================= Dataloader ==========================
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers in dataloader.")
parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches loaded in advance by each worker.")
