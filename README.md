# Modeling Temporal Concept Receptive Field Dynamically for Untrimmed Video Analysis

This is a PyTorch implementation of the model described in our paper:

> Z. Qi, S. Wang, C. Su, L. Su, W. Zhang, and Q. Huang. Modeling Temporal Concept Receptive Field Dynamically for Untrimmed Video Analysis. ACM MM 2020.

# Dependencies

  * Pytorch 1.2.0
  * Cuda 9.2.148
  * Cudnn 7.6.2
  * Opencv-python 4.2.0.34
  * Python 3.6.9 

# Data

## Dataset Prepare

1. Download the pre-trained concept detector weights from [Baidu](https://pan.baidu.com/s/1pihbuOLvML6CrbGrZLFxdg) passward 'wv0e' or [Google Grive](https://drive.google.com/drive/folders/1jPssDmtePpwiJEIdZtNqM6PlLeo5uSyo) and put them in folder weights/

2. Download the FCVID dataset from http://bigvid.fudan.edu.cn/FCVID/.

3. The annotation information of each  dataset is provided in folder data/FCVID/video_labels.

4. Extract the video frames for each video and put the extracted frames in folder data/FCVID/frames/.

   For ActivityNet dataset ( http://activity-net.org/. ) , we use the latest released version of the dataset (v1.3).

# Train

* python main.py --gpu_ids 0,1 --model_name tdcmn_si_soa --dataset FCVID  --no_test

  for other hyperparameters, please refer to opts.py file.

# Test

* Pretrained model weigths are avaiable in [Baidu](https://pan.baidu.com/s/1_I040UUw7vzOKBzqK-PRmg) passward 'szlk' or [Google Grive](https://drive.google.com/drive/folders/1k_TXTMj070FZ446Dn8ZQDGmN1HWfqm0U) 
* Download the pre-trained weights and put them in folder results/

* python main.py --gpu_ids 0,1 --model_name tdcmn_si_soa --dataset FCVID  --resume_path pretrained_model/tdcmn_si_soa.pth --no_train --test_crop_number 1


# **Citation**

Please cite our paper if you use this code in your own work:

```
@inproceedings{qi2020modeling,
  title={Modeling Temporal Concept Receptive Field Dynamically for Untrimmed Video Analysis},
  author={Qi, Zhaobo and Wang, Shuhui and Su, Chi and Su, Li and Zhang, Weigang and Huang, Qingming},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={3798--3806},
  year={2020}
}
```

# Contcat

If you have any problem about our code, feel free to contact

- [zhaobo.qi@vipl.ict.ac.cn](mailto:zhaobo.qi@vipl.ict.ac.cn)

