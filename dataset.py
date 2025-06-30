"""
One-shot Cross-domain Instance Detection With Universal Representation

Data loader
June 29th, 2025
"""


import os
import math
import h5py
import glob
from test import test
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from os.path import join as ospj
from os.path import basename as ospb
from os.path import dirname as ospd

preprocess = transforms.Compose(
    [
        transforms.ToTensor(),  # transforms.ToTensor() transforms the input within [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Define the image transformation
preprocess_swinT = transforms.Compose(
    [
        transforms.Resize(
            (224, 224)
        ),  # Resize the image (Swin Transformer expects a certain size)
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize the tensor
    ]
)

preprocess_clip = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


def load_image(path, out_type=None, resize=None):
    img_raw = Image.open(path).convert("RGB")

    if resize:
        img_raw = img_raw.resize((resize, resize))

    if out_type == "tensor":
        img_raw = preprocess(img_raw)
        img_raw = img_raw.unsqueeze(0)

    return img_raw


class listDataset_OCID_UR(Dataset):
    def __init__(self, args, test_only=False, qualitative=False):
        self.init_seed = False

        # img resize
        self.imrs_size = args["img_resize"]

        # training set augmentation
        aux_fts_folder = "aux-feature"
        self.aug_list = args["data.aug_list"]

        if args["img_resize"]:
            aux_fts_folder += "-resize_" + str(self.imrs_size)
        train_fts_path = "{}/{}/randnum_{:d}_aug".format(
            args["data_root"], aux_fts_folder, args["rand_num"]
        )

        print("Load training set in ", train_fts_path)

        train_npy = glob.glob(ospj(train_fts_path, "*.npy"))
        train_npy.sort()

        train_dict = {}
        for ele in train_npy:
            key = ospb(ele).split(".")[0]
            if key not in train_dict.keys():
                train_dict[key] = []

            train_dict[key].append(ele)

        train_npy = []

        # to extract test data
        fts_folder = args["fts_folder"]
        if args["img_resize"]:
            fts_folder += "-resize_" + str(self.imrs_size)
        temp_exmp = glob.glob(ospj(args["data_root"], "exemplar", fts_folder, "*.npy"))
        temp_sear = glob.glob(
            ospj(args["data_root"], "search_image", fts_folder, "*.npy")
        )
        temp_exmp.sort()
        temp_sear.sort()

        test_dict = {}
        for ele in temp_exmp:
            # ensure the training and test sets are not overlapped
            # example for test_dict: key-001a_exmp，value-[001a-001,001a-002,001a-003]
            key = ospb(ele).split("-")[0] + "_exmp"
            if key not in test_dict.keys():
                test_dict[key] = []
            test_dict[key].append(ele)
        for ele in temp_sear:
            key = ospb(ele).split("-")[0] + "_sear"
            if key not in test_dict.keys():
                test_dict[key] = []
            test_dict[key].append(ele)

        temp_exmp, temp_sear, test_npy = [], [], []

        sear_raw = glob.glob(
            ospj(args["data_root"], "search_image", "*.jpg")
        )
        sear_raw.sort()

        # based on the number of search images，construct training vs. val as 3:7
        self.dataset_size = len(sear_raw)
        self.trainset_size = math.ceil(self.dataset_size * 0.3)

        processed = 0
        train_npy = {"exmp_domain": [], "sear_domain": []}
        train_aug_npy = {"exmp_domain": [], "sear_domain": []}
        test_npy = {"exmp_domain": [], "sear_domain": []}

        random_list_path = ospj(os.getcwd(), "rand_list")
        if not os.path.exists(random_list_path):
            os.mkdir(random_list_path)
        random_list_path = ospj(
            random_list_path,
            "rand_num_"
            + str(args["rand_num"]).zfill(4)
            + "_case_"
            + str(args["case"])
            + ".npy",
        )
        if os.path.exists(random_list_path):
            print("load division file", random_list_path)
            random_list = np.load(random_list_path)
        else:
            print("divide dataset...", end=", ")
            random_list = torch.randperm(self.dataset_size)
            np.save(random_list_path, random_list)

        training_img_num = 0
        for img_i in random_list.tolist():
            ind = ospb(sear_raw[img_i])[:3]
            processed += 1
            if processed < self.trainset_size + 1:
                if test_only is True:
                    continue
                training_img_num += 1
                train_npy["exmp_domain"] += train_dict[ind + "a-ori"]
                train_npy["sear_domain"] += train_dict[ind + "d-ori"]

                for aug_ in self.aug_list:
                    train_aug_npy["exmp_domain"] += train_dict[ind + "a-" + aug_]
                    train_aug_npy["sear_domain"] += train_dict[ind + "d-" + aug_]
            else:
                test_npy["exmp_domain"] += test_dict[ind + "a_exmp"]
                test_npy["sear_domain"] += test_dict[ind + "a_sear"]

        print("training images num: ", training_img_num)

        train_npy["exmp_domain"] += train_aug_npy["exmp_domain"]
        train_npy["sear_domain"] += train_aug_npy["sear_domain"]

        train_dict, test_dict, train_aug_npy = [], [], []
        assert len(train_npy["exmp_domain"]) == len(
            train_npy["sear_domain"]
        ), "wrong training set!"
        assert len(test_npy["exmp_domain"]) == len(
            test_npy["sear_domain"]
        ), "wrong test set!"

        # load training data
        if test_only is False:
            print("load training features...", end=", ")
            self.train_exmp_fts, self.train_sear_fts = [], []
            for npy_i in range(len(train_npy["exmp_domain"])):
                exmp_fts_ = torch.from_numpy(
                    np.load(train_npy["exmp_domain"][npy_i], allow_pickle=True)
                )
                sear_fts_ = torch.from_numpy(
                    np.load(train_npy["sear_domain"][npy_i], allow_pickle=True)
                )
                self.train_exmp_fts.append(exmp_fts_)
                self.train_sear_fts.append(sear_fts_)

            self.train_exmp_fts = torch.cat(self.train_exmp_fts, dim=0).permute(
                0, 1, 3, 2
            )
            self.train_sear_fts = torch.cat(self.train_sear_fts, dim=0).permute(
                0, 1, 3, 2
            )

        # load test data
        # calculate stri, strj for inverser projection
        mat_str = np.zeros((len(test_npy["exmp_domain"]), 2))
        mat_w_h = np.zeros((len(test_npy["exmp_domain"]), 6))
        num_i, num_j = 20, 20

        test_ind = []
        test_num = -1
        for exmp_npy_path in test_npy["exmp_domain"]:
            test_num += 1
            test_ind.append(int(ospb(exmp_npy_path).split("-")[1].split(".")[0]) - 1)

            exmp_jpg_path = exmp_npy_path.replace(fts_folder, "").replace(
                ".npy", ".jpg"
            )

            sear_jpg_path = ospj(
                ospd(exmp_jpg_path).replace("exemplar", "search_image"),
                ospb(exmp_jpg_path).split("-")[0][:-1] + "d.jpg",
            )
            tmp = load_image(exmp_jpg_path)
            sea = load_image(sear_jpg_path)
            t_w, t_h = tmp.size[:2]
            I_w, I_h = sea.size[:2]

            if args["img_resize"]:
                # from index to boxes & inverse_projection
                inv_w, inv_h = self.imrs_size, self.imrs_size
            else:
                inv_w, inv_h = t_w, t_h
            mat_w_h[test_num, :] = [
                I_w,
                I_h,
                t_w,
                t_h,
                inv_w,
                inv_h,
            ]

            if args["img_resize"]:
                I_w, I_h = round(I_w * self.imrs_size / t_w), round(
                    I_h * self.imrs_size / t_h
                )
                t_w, t_h = self.imrs_size, self.imrs_size

            mat_str[test_num, 0] = math.floor((I_h - t_h) / (num_i - 1))  # str_i
            mat_str[test_num, 1] = math.floor((I_w - t_w) / (num_j - 1))  # str_j

        # load ground truth
        test_gt_path = ospj(args["data_root"], "GroundTruth-" + args["case"] + ".mat")
        gt_mat = h5py.File(test_gt_path)
        gt_mat = np.transpose(gt_mat["bndboxLoc"])
        gt_mat = gt_mat[: self.dataset_size * 3, :]
        gt_mat = gt_mat[test_ind, :]

        # load test data
        print("load test features...", end="\r")
        test_exmp_fts, test_sear_fts = [], []
        test_exmp_npy_name = []
        for npy_i in range(len(test_npy["exmp_domain"])):
            test_exmp_fts.append(
                torch.from_numpy(
                    np.load(test_npy["exmp_domain"][npy_i], allow_pickle=True)
                )
            )
            test_sear_fts.append(
                torch.from_numpy(
                    np.load(test_npy["sear_domain"][npy_i], allow_pickle=True)
                )
            )

            test_exmp_npy_name.append(ospb(test_npy["exmp_domain"][npy_i]))

        test_exmp_fts = torch.cat(test_exmp_fts, dim=0).permute(0, 1, 3, 2)
        test_sear_fts = torch.cat(test_sear_fts, dim=0).permute(0, 1, 3, 2)

        # data conclusion
        self.test_data = {
            "gt": gt_mat,
            "str": mat_str,
            "w_h": mat_w_h,
            "exmp": test_exmp_fts,
            "sear": test_sear_fts,
        }

        self.test_data["exmp_name"] = test_exmp_npy_name

        if test_only is False:
            self.pos_pair_num, self.bb_sel_num, self.feature_map_size, self.fts_dim = (
                self.train_exmp_fts.shape
            )

            self.sample_times = self.pos_pair_num
            self.sample_times /= len(self.aug_list) + 1

    def __len__(self):
        self.sample_times = int(self.sample_times)
        return self.sample_times

    def __getitem__(self, index):
        # each output feature is of size 1 * backbone_num * fts_dim
        assert index <= len(self), "index range error"

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        ori_smpl_num = self.sample_times
        crop_smpl_num = 400  # randomly cropped 400 samples each image
        # start = index + ori_smpl_num
        start = (
            index
            + ori_smpl_num
            + crop_smpl_num * len(self.aug_list) * math.floor(index / crop_smpl_num)
        )
        pos_idx_collect = [index] + [
            start + crop_smpl_num * i for i in range(len(self.aug_list))
        ]

        td_anc = self.train_exmp_fts[pos_idx_collect]
        sd_pos = self.train_sear_fts[pos_idx_collect]
        # print(pos_idx_collect)
        return td_anc, sd_pos
