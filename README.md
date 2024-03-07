# SHDA

This repository contains source codes necessary to reproduce the results presented in "Noises are Transferable: What is the Transferable Knowledge in Semi-supervised Heterogeneous Domain Adaptation?". In addition, to the best of our knowledge, it is the first relatively comprehensive SHDA open-source repository, which can be used to further inspire promising research. 

# Approaches

Currently, we summarize seven SHDA approaches in the Baselines folder, i.e., SHFA [1], CDLS [2], DDACL [3], TNT [4], STN [5], SSAN [6], and JMEA [7], based on their open-source codes. In addition, we also include two supervised learning methods, i.e., SVMt and NNt.

1. [SHFA](https://github.com/wenli-vision/SHFA_release) (Semi-supervised Heterogeneous Feature Augmentation) [1]
2. [CDLS](https://github.com/yaohungt/CrossDomainLandmarksSelectionCDLS/tree/master) (Cross-Domain Landmark Selection ) [2]
3. [DDACL](https://github.com/yyyaoyuan/DDA) (Discriminative Distribution Alignment with Cross-entropy Loss) [3]
4. [TNT](https://github.com/wyharveychen/TransferNeuralTrees) (Transfer Neural Trees) [4]
5. [STN](https://github.com/yyyaoyuan/STN) (Soft Transfer Network) [5]
6. [SSAN](https://github.com/BIT-DA/SSAN) (Simultaneous Semantic Alignment Network) [6]
7. [JMEA](https://github.com/fang-zhen/Semi-supervised-Heterogeneous-Domain-Adaptation) (Joint Mean Embedding Alignment ) [7]
8. [SVMt](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) (Support Vector Machine) [8]
9. NNt (a Simple Neural Network)

In addition, we have open-sourced up our noise generation codes in the Noise_Generation folder.

# Datasets

In our experiments, we adopt three widely-used datasets: Office+Caltech-10 [9], [10], Multilingual Reuters Collection [11], and NUS-WIDE+ImageNet-8 [12], [13]. In addition, we also expose the noise datasets constructed by ourselves.
All of the datasets can be downloaded from https://drive.google.com/drive/folders/15fTKaA0m0aT-7MT19khcyMNCyXUGT8nA?usp=drive_link. You can unzip those folders and put them under the Datasets folder.


# Installation

1. SHFA, CDLS, DDACL, TNT, and SVMt have been implemented using MATLAB, you need to install MATLAB to run them. 

2. STN, JMEA, and NNT have been implemented using the Tensorflow framework [14], you need to follow the steps below to configure the running environment, please see details in [STN](https://github.com/yyyaoyuan/STN) .

```
conda create -n runenv python=3.6
conda activate runenv

pip install tensorflow-gpu==1.4
conda install cudatoolkit=8.0
conda install cudnn=6.0
conda install scipy
conda install matplotlib
conda install scikit-learn
```

3. SSAN has been implemented using the Pytorch framework [15], you need to follow the steps below to configure the running environment, please see details in [SSAN](https://github.com/BITDA/SSAN) .

```
conda create -n ssan -y python=3.6
conda activate ssan
conda install -y ipython pip
pip install torch==1.3.1       or  conda install torch==1.3.1
pip install matplotlib==3.1.2
pip install scipy==1.3.2
pip install numpy==1.17.4
pip install scikit_learn==0.22.1
```

 
# Usage

Given that the data-loading modules within the open-source codes of various baselines are distinct, we standardize those modules into a unified format. This allows for the flexible loading of experimental datasets by editing the corresponding configuration files, i.e., the **add_dependencies.py / add_dependencies.m** file in various methods.

Thus, you just need to edit the **add_dependencies.py / add_dependencies.m** file to set different datasets. Also, you can edit **main.py / main.m** file to load different datasets except for SSAN. The run commands for all methods are summarized as follows.

1. For SHFA, CDLS, DDACL, TNT, and SVMt, when you enter the home directory of the corresponding method, please run:
```
nohup matlab -nodesktop -nosplash -r main > run.log 2>&1 &
```

2. For STN, JMEA, and NNT, when you enter the home directory of the corresponding method, please run:
```
nohup python main.py > run.log 2>&1 &
```

3. For SSAN, when you enter the home directory, please run:
```
nohup python main.py --source ND100 --target TS5 --cuda 0 --nepoch 1000 --partition 10 --prototype three --layer double --d_common 256 --optimizer mSGD --lr 0.1 --alpha 0.1 --beta 0.004 --gamma 0.1 --combine_pred Cosine --checkpoint_path checkpoint/ --temperature 5.0 > run_ND100TS5.log 2>&1 &
```
Note that, for SSAN, you first need to edit the add_dependencies.py and add_path.py to set different datasets. Then, you can modify the parameters of --source and -- target to load different datasets. 

# Acknowledgements

We would like to express our deep gratitude to the authors of all SHDA approaches utilized in this paper, and we sincerely thank them for open-sourcing their codes to promote the development of the SHDA field.  In addition, if you have any licensing issues, please feel free to contact yaoyuan(dot)hitsz(at)gmail(dot)com.

# References

[1] W. Li, L. Duan, D. Xu, and I. W. Tsang, “Learning with augmented features for supervised and semi-supervised heterogeneous domain adaptation,” TPAMI, vol. 36, no. 6, pp. 1134–1148, 2014.

[2] Y. H. H. Tsai, Y. R. Yeh, and Y. C. F. Wang, “Learning cross-domain landmarks for heterogeneous domain adaptation,” in CVPR, 2016, pp.5081–5090.

[3] Y. Yao, Y. Zhang, X. Li, and Y. Ye, “Discriminative distribution alignment: A unified framework for heterogeneous domain adaptation,” Pattern Recognition, vol. 101, p. 107165, 2020.

[4] W.-Y. Chen, T.-M. H. Hsu, Y.-H. Tsai, Y.-C. F. Wang, and M.-S. Chen, “Transfer neural trees for heterogeneous domain adaptation,” in ECCV, 2016.

[5] Y. Yao, Y. Zhang, X. Li, and Y. Ye, “Heterogeneous domain adaptation via soft transfer network,” in ACM MM, 2019, p. 1578–1586.

[6] S. Li, B. Xie, J. Wu, Y. Zhao, C. H. Liu, and Z. Ding, “Simultaneous semantic alignment network for heterogeneous domain adaptation,” in ACM MM, 2020, p. 3866–3874.

[7] Z. Fang, J. Lu, F. Liu, and G. Zhang, “Semi-supervised heterogeneous domain adaptation: Theory and algorithms,” TPAMI, vol. 45, no. 1, pp. 1087–1105, 2023.

[8] C.-C. Chang and C.-J. Lin, “Libsvm: a library for support vector machines,” ACM TIST, vol. 2, no. 3, pp. 1–27, 2011

[9] K. Saenko, B. Kulis, M. Fritz, and T. Darrell, “Adapting visual category models to new domains,” in ECCV, 2010, pp. 213–226.

[10] G. Griffin, A. Holub, and P. Perona, “Caltech-256 object category dataset,” California Institute of Technology, Tech. Rep. 7694, 2007

[11] M. Amini, N. Usunier, and C. Goutte, “Learning from multiple partially observed views - an application to multilingual text categorization,” in NeurIPS, 2009, pp. 28–36.

[12] T.-S. Chua, J. Tang, R. Hong, H. Li, Z. Luo, and Y. Zheng, “Nus-wide: A real-world web image database from national university of singapore,” in CIVR, 2009, pp. 48:1–48:9.

[13] J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. Fei-Fei, “Imagenet: A large-scale hierarchical image database,” in CVPR, 2009, pp. 248–255.

[14] M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard, M. Kudlur, J. Levenberg, R. Monga, S. Moore, D. G. Murray, B. Steiner, P. Tucker, V. Vasudevan, P. Warden, M. Wicke, Y. Yu, and X. Zheng, “Tensorflow: A system for large-scale machine learning,” in OSDI, 2016, pp. 265–283.

[15] Paszke A, Gross S, Massa F, et al. Pytorch: An imperative style, high-performance deep learning library. in NeurIPS, 2019, 32.
