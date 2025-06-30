"""
One-shot Cross-domain Instance Detection With Universal Representation

Train model UIIDCL
June 29th, 2025
"""

import os
import time
import random
import logging
import numpy as np
from os.path import join as ospj

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributions
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from test import test
from config import args
from models import OCID_UR_Net, AverageMeter
from dataset import listDataset_OCID_UR
from model_utils import (
    save_checkpoint,
    load_checkpoint,
    load_last_checkpoint,
    UIIDCL_BatchHard,
    model_name_generation,
    check_dir,
)

from torch.autograd import Variable

# prepare environment
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu"])[1:-1]
torch.cuda.empty_cache()

# model name generation
model_name, modelname_suffix = model_name_generation(args)

code_root = os.getcwd()
case_flag = "Case-" + args["case"]
path_to_save_model = ospj(code_root, "result-FUR", case_flag, "model")
path_to_save_log = ospj(code_root, "result-FUR", case_flag, "log")
check_dir(path_to_save_model)
check_dir(path_to_save_log)


def train():
    # model initialization
    model = OCID_UR_Net(args).to(device)

    # log
    start_epoch = 0
    log_file_path = ospj(path_to_save_log, model_name + ".log")

    if os.path.exists(log_file_path):
        logging.basicConfig(
            format="%(asctime)s %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            filename=log_file_path,
            level=logging.INFO,
        )
        input = open(log_file_path, "r")
        for line in input:
            if "Epoch#" in line:
                start_epoch = (
                    int(line[line.find("Epoch#") + 6 : line.find(" (bs=")]) + 1
                )
                model = load_last_checkpoint(model, path_to_save_model, model_name)
            if "mean_mAP" in line or start_epoch > args["model.total_epoch"]:
                if not args["test_only"]:
                    return print(args["data_root"].split("/")[-1], ", already trained.")
    else:
        logging.basicConfig(
            format="%(asctime)s %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            filename=log_file_path,
            level=logging.INFO,
        )
        logging.info("=" * 30)
        for k in list(args.keys()):
            logging.info("%s: %s" % (k, args[k]))
        logging.info("=" * 30)

    # data loader
    ocid_data = listDataset_OCID_UR(args)
    train_loader = torch.utils.data.DataLoader(
        ocid_data, batch_size=args["model.bs"], num_workers=args["num_workers"]
    )

    # test data loader
    gt_mat, mat_str, mat_w_h, exmp, sear = (
        ocid_data.test_data["gt"],
        ocid_data.test_data["str"],
        ocid_data.test_data["w_h"],
        ocid_data.test_data["exmp"],
        ocid_data.test_data["sear"],
    )

    UIIDCL_bh = UIIDCL_BatchHard(margin=args["model.UIIDCL_bh_margin"]).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["model.init_lr"], weight_decay=args["model.decay"]
    )

    # Schedule learning rate
    def adjust_lr(epoch):
        lr = (
            args["model.init_lr"]
            if epoch <= 100
            else args["model.init_lr"] * (0.001 ** ((epoch - 100) / 50.0))
        )
        for g in optimizer.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1)

    # test-only
    if args["test_only"]:
        print("test...")
        model = load_checkpoint(model, path_to_save_model, model_name)
        if len(args["gpu"]) > 1:
            model = model.module
        model.eval()
        mAP = test(model, gt_mat, mat_str, mat_w_h, exmp, sear, device_=device)
        test_info = "test mean mAP={:.3f}".format(mAP)
        print(test_info)
        logging.info(test_info)
        return

    # train model
    loss_stat = AverageMeter()

    for epoch in range(args["model.total_epoch"]):
        if epoch < start_epoch:
            continue
        adjust_lr(epoch)

        model.train()
        train_start = time.time()
        i_stop_step, mAP = 0, []
        for i, (exmp_anc, sear_pos) in enumerate(train_loader):
            exmp_anc = exmp_anc.to(device)
            sear_pos = sear_pos.to(device)
            print("training_data_size: ", exmp_anc.shape)
            if exmp_anc.shape[0] <= 1:
                continue

            train_start_step_t = time.time()
            exmp_label = torch.from_numpy(np.arange(exmp_anc.shape[0])).long().to(device)
            lbl = torch.cat((exmp_label, exmp_label))

            exmp_anc_, sear_pos_, lbl_ = (
                torch.tensor([]).to(device),
                torch.tensor([]).to(device),
                torch.tensor([]).to(device),
            )
            for index in range(len(ocid_data.aug_list) + 1):
                exmp_anc_ = torch.cat((exmp_anc_, exmp_anc[:, index, :, :, :]), dim=0)
                sear_pos_ = torch.cat((sear_pos_, sear_pos[:, index, :, :, :]), dim=0)
                lbl_ = torch.cat((lbl_, exmp_label, exmp_label))
            exmp_anc, sear_pos, lbl = exmp_anc_, sear_pos_, lbl_

            t1 = time.time()
            (
                exmp_anc,
                sear_pos,
            ) = model(exmp_fts=exmp_anc.to(device), sear_fts=sear_pos.to(device))

            model_forward_t = time.time() - t1
            exmp_norm = torch.pow(exmp_anc, 2).sum(dim=-1).sqrt().float()

            t1 = time.time()

            tri_loss, tri_prec, tri_num = UIIDCL_bh(
                torch.cat((exmp_anc, sear_pos), 0), lbl
            )

            loss = tri_loss
            loss_stat.update(tri_loss.item())
            model_loss_cal_t = time.time() - t1

            t1 = time.time()

            optimizer.zero_grad()
            loss.backward()  # retain_graph=True
            optimizer.step()
            model_backprop_t = time.time() - t1

            train_end = time.time()

            print_loss = "tri_loss={:.2f}, acc={:.2f}({:d})".format(
                loss_stat.avg, tri_prec, tri_num
            )

            print(
                "Ep[{}][{}/{}]{}, norm:{:.2f},{:.2f}, {}, gpu:{}".format(
                    epoch,
                    i + 1,
                    len(train_loader),
                    args["data_root"].split("/")[
                        -1
                    ],  # lr=optimizer.state_dict()['param_groups'][0]['lr'],
                    exmp_norm.mean().item(),
                    exmp_norm.std().item(),
                    print_loss,
                    str(args["gpu"]),
                )
            )
            print(
                "    forward: {t1:.1f}s, loss_cal: {t2:.1f}s, backprop: {t3:.1f}s, total:{t4:.1f}s".format(
                    t1=model_forward_t,
                    t2=model_loss_cal_t,
                    t3=model_backprop_t,
                    t4=train_end - train_start_step_t,
                )
            )

            i_stop_step += 1
            if i_stop_step >= args["model.maxstep_per_epoch"]:
                break

        train_end_epoch = time.time()
        print(
            "TIME FOR Training one epoch:{:.2f}".format(train_end_epoch - train_start)
        )

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            path_to_save_model,
            model_name,
            is_best=True,
        )

        print_mAP = ""
        train_test_info = "Epoch#{:03d} (bs={:03d}, step:{:04d}): {}, train_time={:.1f}s, exmp_norm={:.2f}(std:{:.2f}, {})".format(
            epoch,
            args["model.bs"],
            i_stop_step,
            print_loss,
            train_end - train_start,
            exmp_norm.mean().item(),
            exmp_norm.std().item(),
            print_mAP,
        )
        print("->" * 20 + print_mAP)
        logging.info(train_test_info)


if __name__ == "__main__":
    # set fixed random number to make fair comparisons
    rand_num = args["rand_num"]
    torch.manual_seed(rand_num)
    np.random.seed(rand_num)
    random.seed(rand_num)
    train()
