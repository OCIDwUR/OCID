# One-shot Cross-domain Instance Detection With Universal Representation

## Introduction

This repository is released for our code and dataset for `One-shot Cross-domain Instance Detection With Universal Representation`. The dataset is contained within the folder  `datasets`, which includes the 6,177 OCID tasks originated from 14 different data sources. The main codes are written and run under `python` platform.

## Preliminary

* Download datasets and backbones from Baidu netdisk. The following link is permanently effective. If you have any questions or other requirements, please do not hesitate to contact me at xx.

  ```
  Link: https://pan.baidu.com/s/1YOZffjc2VdFRk2DaKy5nOg?pwd=OCID 
  Code: OCID
  ```

  There are 14 data sources, each of which contains folders as follows:

  | folder                | content                                           |
  | --------------------- | ------------------------------------------------- |
  | exemplar              | exemplars                                         |
  | exemplar_source_image | the source images where the exemplars are cropped |
  | search_image          | search images with image degradations             |
  | search_image_original | original clear search images                      |

* Prepare experimental environment following `requirements.txt`.

  ```
  pip install -r requirements.txt
  ```

* We recommend pre-extracting the frozen backbone features to accelerate the training and test speed.

  1) Extract backbone features for exemplars and samples from search images (case A as an example).

  ```
  python pre-extract-fts-base -case A 
  ```

  2) Data Augmentation and extract backbone features for the augmented (case A as an example).

  ```
  python pre-extract-fts-aug -case A
  ```

## Training and Test

* run demo.py. Please refer to the `config.py` for the setting of optional arguments.

  1) Training (case A as an example).

  ```
  python demo.py -case A
  ```

  2) Test  (case A as an example).

  ```
  python demo.py -case A -test_only
  ```

## Other Applications

Based on the reviewer's insightful suggestion, our datasets can be applied to the research on other tasks, especially the task of cross-domain fine-grained instance detection following the setting of domain adaptation in image classification.  Specifically, it supports the domain adaptation setting where both exemplars (source domain) and corresponding unlabeled search images (target domain) are provided during the training stage.  

## Acknowledgements

Thank Lu Liu et al. for their proposition and implementation of Universal Representation (for single-domain coarse-level image classification). They help and inspire this work.

## Bibtex

```
@article{OCID,
title = {One-shot Cross-domain Instance Detection With Universal Representation}
}
```
