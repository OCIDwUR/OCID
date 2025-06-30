import os
import torch
import argparse

from os.path import join as ospj

parser = argparse.ArgumentParser(
    description="One-shot Instance Detection With Universal Representation"
)

# model
parser.add_argument("--model.ASNet_hidden_dim", type=float, default=256)
parser.add_argument("--model.backbone", "-backbone", default="resnet18", type=str)
parser.add_argument("--model.bs", "-bs", type=int, default=32)

parser.add_argument(
    "--model.dropout",
    type=float,
    default=0,
    help="Adding dropout inside a basic block of widenet",
)
parser.add_argument(
    "--model.fts_dim",
    type=float,
    default=0,
    help="dimension of universal representation",
)
parser.add_argument("--model.init_lr", type=float, default=2e-4)
parser.add_argument("--model.decay", type=float, default=5e-4)

parser.add_argument("--model.total_epoch", "-epoch", type=int, default=10)
parser.add_argument(
    "--model.maxstep_per_epoch", "-step", type=int, default=1000
)  # 1000
parser.add_argument(
    "--model.UIIDCL_bh_margin",
    "-bh_m",
    type=float,
    default=0.8,
    help="margin=0 denotes soft margin",
)
parser.add_argument("--model.dropout_en", "-dropout", action="store_true")
parser.add_argument('--model.init', '-init', action='store_true')
parser.add_argument("--num_workers", "-nw", type=int, default=4) 
parser.add_argument(
    "--gpu",
    "-gpu",
    metavar="GPUs",
    type=int,
    nargs="+",
    default=[0],
    help="multiple gpu",
)

# case
parser.add_argument(
    "--data_root", default=None, type=str, help="dataset root; do not need setting"
)
parser.add_argument(
    "--case", "-case", default="A", type=str, choices=list("ABCDEFGHIJKLMN")
)
parser.add_argument("--fts_layer", "-fts_layer", default="conv4", type=str)

parser.add_argument("--test_only", "-test_only", action="store_true")
parser.add_argument("--fts_folder", "-fts_folder", default="feature", type=str)

# others
parser.add_argument(
    "--rand_num", "-rn", default=1234, type=int, help="random control number"
)
parser.add_argument(
    "--img_resize",
    "-imrs",
    default=50,
    type=int,
    help="the size after img_resize; None means no resize.",
)

args = vars(parser.parse_args())

code_root = os.getcwd()
args["data_root"] = ospj(code_root, "datasets", "DataSource" + args["case"])
args["exemplar_folder"] = "exemplar"
args["sear_img_folder"] = "search_image"

DATASET_MODELS_RESNET18 = {
    "ilsvrc_2012": "imagenet-net",
    "omniglot": "omniglot-net",
    "aircraft": "aircraft-net",
    "cu_birds": "birds-net",
    "dtd": "textures-net",
    "quickdraw": "quickdraw-net",
    "fungi": "fungi-net",
    "vgg_flower": "vgg_flower-net",
}

assert args["fts_layer"] == "conv4"
args["model.fts_dim"] = 512  # feature dimension
args["model.backbone_num"] = 8  # number of backbones
args["model.fm_size"] = 9  # size of feature map
args["data.aug_list"] = ["blur", "mask", "color", "mixup", "mir"]  # augmentation type
