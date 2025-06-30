"""
One-shot Cross-domain Instance Detection With Universal Representation

Data Augmentation and extract backbone features for the augmented.
June 29th, 2025
"""

import os
import glob
import torch
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
from os.path import join as ospj
from os.path import basename as ospb
from model_utils import check_dir
from dataset import load_image, preprocess
from models import get_multinet_extractor
from config import args


# prepare environment
os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu"])[1:-1]
torch.cuda.empty_cache()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Python 中设置

# initialize model
embed_many = get_multinet_extractor(args)
print("")
print("loaded all models.")

# img resize
imrs_size = args["img_resize"]

# folder set
folder_in_work = ["exemplar_source_image", "search_image"]

# augmentation type (refer to the papers for detailed statement)
aug_list = ["blur", "mask", "color", "mixup", "mir"]


def process_aux_original_scale_imrs():

    # collect image addresses
    image_list = {}
    for domain in folder_in_work:
        image_list[domain] = glob.glob(ospj(args["data_root"], domain, "*.jpg"))
        image_list[domain].sort()

    # feature
    fts_path = "{}/aux-feature-resize_{:d}/randnum_{:d}_aug".format(
        args["data_root"], imrs_size, args["rand_num"]
    )
    check_dir(fts_path)

    # start cropping 20*20 patches and extracting features
    crop_num, img_raw, img_aug = 20, {}, {}

    total_num = len(image_list["exemplar_source_image"])
    for img_i in range(total_num):
        print(args["data_root"].split("dataset")[-1], end=" | ")
        for domain in image_list.keys():
            print(ospb(image_list[domain][img_i]), end=" , ")
            img_raw[domain], img_aug[domain] = [], {}
        print("num: ", img_i + 1, "/", total_num, end=", ")

        image, w, h = {}, [], []
        for domain in image_list.keys():
            image[domain] = load_image(image_list[domain][img_i])
            img_w, img_h = image[domain].size[:2]
            w.append(img_w)
            h.append(img_h)
        w, h = set(w), set(h)
        assert len(w) == len(h) == 1, "wrong!"

        rand_row = random.sample(range(img_h), crop_num)
        rand_col = random.sample(range(img_w), crop_num)

        raw_shape_set, aug_shape_set = [], []
        for domain in image_list.keys():
            image_name = ospb(image_list[domain][img_i])
            save_raw_fts_name = ospj(fts_path, image_name[:-4] + "-ori.npy")

            if os.path.exists(save_raw_fts_name):
                continue

            for aug_type in aug_list:
                img_aug[domain][aug_type] = []

            for patch_i in range(crop_num):
                for patch_j in range(crop_num):
                    sta_row, sta_col = [rand_row[patch_i], rand_col[patch_j]]
                    img_temp = image[domain].crop(
                        (sta_col, sta_row, sta_col + imrs_size, sta_row + imrs_size)
                    )
                    img_raw[domain].append(preprocess(img_temp).unsqueeze(0))
                    for aug_type in aug_list:
                        img_aug[domain][aug_type].append(
                            preprocess(
                                image_augmentation(img_temp, aug_type)
                            ).unsqueeze(0)
                        )

            raw_shape = extract_and_save_fts(
                torch.cat(img_raw[domain], dim=0), save_raw_fts_name
            )
            for aug_type in aug_list:
                aug_shape = extract_and_save_fts(
                    torch.cat(img_aug[domain][aug_type], dim=0),
                    save_raw_fts_name.replace("ori", aug_type),
                )
                img_aug[domain][aug_type] = []
            img_raw[domain], img_aug[domain] = [], {}
            raw_shape_set.append(list(raw_shape))
            aug_shape_set.append(list(aug_shape))

        if raw_shape_set != []:
            assert (
                np.linalg.matrix_rank(np.array(raw_shape_set)) <= 1
            ), "Wrong Feature Extraction for RAW!"
            assert (
                np.linalg.matrix_rank(np.array(aug_shape_set)) <= 1
            ), "Wrong Feature Extraction for AUG!"
            assert list(raw_shape) == list(aug_shape), "RAW and AUG not Consistent!"
            print("FtsShape:", list(raw_shape))
        else:
            print("")


def image_augmentation(img, type):
    assert type in ["mir", "blur", "mask", "color", "mixup"]

    rand_num = random.sample([1, 2, 3], 1)[0]
    if type == "mir":
        output = img.transpose(Image.FLIP_LEFT_RIGHT)
    if type == "blur":
        if rand_num == 1:
            output = img.filter(ImageFilter.BLUR)
        if rand_num == 2:
            output = img.filter(ImageFilter.GaussianBlur)
        if rand_num == 3:
            output = img.filter(ImageFilter.BoxBlur(5))
    if type == "mask":
        draw = ImageDraw.Draw(img)
        w, h = img.size[:2]

        box_w, box_h = int(w / 10), int(h / 10)
        x1 = random.sample(range(w - box_w), 1)[0]
        y1 = random.sample(range(h - box_h), 1)[0]
        x2 = x1 + box_w
        y2 = y1 + box_h
        axis_info = [x1, y1, x2, y2]
        if rand_num == 1:
            draw.rectangle(axis_info, fill="black")
        if rand_num == 2:
            draw.chord(axis_info, 0, 270, fill="black")
        if rand_num == 3:
            draw.ellipse(axis_info, fill="black")
        output = img
    if type == "color":
        if rand_num == 1:
            enh_img = ImageEnhance.Color(img)
            output = enh_img.enhance(factor=1.4)
        if rand_num == 2:
            enh_img = ImageEnhance.Contrast(img)
            output = enh_img.enhance(factor=1.4)
        if rand_num == 3:
            enh_img = ImageEnhance.Brightness(img)
            output = enh_img.enhance(factor=1.4)
    if type == "mixup":
        red_mask = Image.new("RGB", img.size, "red")
        output = Image.blend(img, red_mask, 0.5)

    return output


def extract_and_save_fts(img_raw, path_for_tensor):
    if args["img_resize"]:
        embed_tensor = []
        for img_chunk in torch.chunk(img_raw, 16, dim=0):
            # embed_tensor.append(embed_many(img_chunk.cuda()))
            embed_tensor.append(embed_many(img_chunk))
        embed_tensor = torch.cat(embed_tensor, dim=0)

    np.save(path_for_tensor, embed_tensor)

    return embed_tensor.shape


if __name__ == "__main__":

    np.random.seed(args["rand_num"])
    random.seed(args["rand_num"])
    process_aux_original_scale_imrs()
