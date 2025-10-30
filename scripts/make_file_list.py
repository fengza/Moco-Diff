import sys
sys.path.append("/public_bme/data/lifeng/code/moco/DiffBIR-main")
#sys.path.append("../..")
import os
from argparse import ArgumentParser
from utils.file import list_image_files

curPath_ = os.path.dirname(__file__)
curPath = os.getcwd()
parser = ArgumentParser()
parser.add_argument("--img_folder", type=str, default='/public_bme/data/lifeng/data/hcp', required=False)
parser.add_argument("--val_size", type=int, default=63, required=False)
parser.add_argument("--save_folder", type=str, default='/public_bme/data/lifeng/data', required=False)
parser.add_argument("--follow_links", action="store_true")
args = parser.parse_args()

files = list_image_files(args.img_folder)

print(f"find {len(files)} images in {args.img_folder}")
assert args.val_size < len(files)

val_files = files[:args.val_size]
train_files = files[args.val_size:]

os.makedirs(args.save_folder, exist_ok=True)

with open(os.path.join(args.save_folder, "train_hcp.list"), "w") as fp:
    for file_path in train_files:
        fp.write(f"{file_path}\n")

with open(os.path.join(args.save_folder, "val_hcp.list"), "w") as fp:
    for file_path in val_files:
        fp.write(f"{file_path}\n")
