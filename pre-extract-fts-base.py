"""
One-shot Cross-domain Instance Detection With Universal Representation

Extract backbone features for exemplars and samples from search images
June 29th, 2025
"""

import os
import math
import time
import torch
import random
import numpy as np

from os.path import join as ospj
from model_utils import check_dir
from dataset import load_image, preprocess
from models import get_multinet_extractor
from config import args

# prepare environment
os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu"])[1:-1]
torch.cuda.empty_cache()

# initialize model
embed_many = get_multinet_extractor(args)
print("")
print("loaded all models.")

# img resize
imrs_size = args["img_resize"]

# folder set
folder_in_work = ["exemplar", "search_image"]


def process_template_and_search():
    # extract and save features
    for image_folder in folder_in_work:
        sub_root = ospj(args["data_root"], image_folder)

        exmp_name_list = os.listdir(ospj(args["data_root"], "exemplar"))
        exmp_name_list.sort()

        fts_folder = "feature"
        if args["img_resize"]:
            fts_folder += "_".join(("-resize", str(args["img_resize"])))
        fts_path = ospj(sub_root, fts_folder)
        print(fts_path)
        check_dir(fts_path)

        img_processed_num = 0
        image_name_list = os.listdir(sub_root)
        image_name_list.sort()
        exmp_shape_set, sear_shape_set = [], []
        for image_name in image_name_list:
            if image_name.split(".")[-1] not in ["bmp", "jpg"]:
                continue

            img_processed_num += 1
            print_name = ospj(sub_root, image_name).split("dataset")[-1] + ": "
            path_for_tensor = ospj(fts_path, image_name.split(".")[0] + ".npy")

            processed = (
                ", processed:"
                + str(img_processed_num)
                + "/"
                + str(len(image_name_list))
            )
            if (image_folder == "exemplar") and (
                os.path.exists(path_for_tensor) is False
            ):
                # extract exemplar features
                img_raw = load_image(
                    ospj(sub_root, image_name),
                    out_type="tensor",
                    resize=args["img_resize"],
                )
                exmp_shape = extract_and_save_fts(img_raw, path_for_tensor)
                exmp_shape_set.append(list(exmp_shape))
                print(
                    print_name,
                    "exemplar, ",
                    os.path.basename(path_for_tensor),
                    processed,
                    ", size:",
                    list(exmp_shape),
                )

            if "search" in image_folder:
                # extract features for samples from the search image
                # read exemplar to resize search image
                for exmp_name in exmp_name_list:
                    path_for_tensor = ospj(fts_path, exmp_name.split(".")[0] + ".npy")
                    if os.path.exists(path_for_tensor):
                        continue

                    img_raw = []
                    judge_str = image_name[:-4].replace("d", "a")
                    if judge_str not in exmp_name:
                        continue
                    exemplar_path = ospj(
                        sub_root.replace("search_image", "exemplar"), exmp_name
                    )
                    exemplar = load_image(exemplar_path)
                    t_w, t_h = exemplar.size[:2]

                    # read search image
                    search_image = load_image(ospj(sub_root, image_name))
                    I_w, I_h = search_image.size[:2]

                    if args["img_resize"]:
                        I_w, I_h = round(I_w * imrs_size / t_w), round(
                            I_h * imrs_size / t_h
                        )
                        search_image = search_image.resize((I_w, I_h))
                        t_w, t_h = imrs_size, imrs_size

                    # split the search image into 20*20 patches uniformly
                    num_i, num_j = [20, 20]
                    str_i = math.floor((I_h - t_h) / (num_i - 1))
                    str_j = math.floor((I_w - t_w) / (num_j - 1))
                    for i in range(num_i):
                        for j in range(num_j):
                            sta_row, sta_col = str_i * i, str_j * j
                            img = search_image.crop(
                                (sta_col, sta_row, sta_col + t_w, sta_row + t_h)
                            )
                            img_raw.append(preprocess(img).unsqueeze(0))
                    # construct batch to speed up network forward propagation
                    img_raw = torch.cat(img_raw, dim=0)
                    sear_shape = extract_and_save_fts(img_raw, path_for_tensor)
                    sear_shape_set.append(list(sear_shape))
                    print(
                        print_name,
                        "search image, ",
                        os.path.basename(path_for_tensor),
                        processed,
                        ", size:",
                        list(sear_shape),
                    )

        rank_exmp = np.linalg.matrix_rank(np.array(exmp_shape_set))
        rank_sear = np.linalg.matrix_rank(np.array(sear_shape_set))
        assert rank_exmp <= 1, "Wrong Feature Extraction for Exmp, rank-" + str(
            rank_exmp
        )
        assert rank_sear <= 1, "Wrong Feature Extraction for Sear, rank-" + str(
            rank_sear
        )


def extract_and_save_fts(img_raw, path_for_tensor):

    embed_tensor = []
    for img_chunk in torch.chunk(img_raw, 16, dim=0):
        # embed_tensor.append(embed_many(img_chunk.cuda()))
        embed_tensor.append(embed_many(img_chunk))
    embed_tensor = torch.cat(embed_tensor, dim=0)

    np.save(path_for_tensor, embed_tensor)

    return embed_tensor.shape


if __name__ == "__main__":
    start_t = time.time()
    np.random.seed(args["rand_num"])
    random.seed(args["rand_num"])

    process_template_and_search()
    end_t = time.time()
    print("test consumed {:.2f}s".format(end_t - start_t))
