# TransFG 复现

## 介绍

我们使用 [Jittor 框架]([Jittor/jittor: Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators.](https://github.com/Jittor/jittor)) 复现了基于 [Torch](https://github.com/pytorch/pytorch) 的 [TransFG]([TACJu/TransFG: This is the official PyTorch implementation of the paper "TransFG: A Transformer Architecture for Fine-grained Recognition" (Ju He, Jie-Neng Chen, Shuai Liu, Adam Kortylewski, Cheng Yang, Yutong Bai, Changhu Wang, Alan Yuille).](https://github.com/TACJu/TransFG)) 项目，在数据集 [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) 和 [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) 上进行了测试，并针对 Patch Split，Contrastive Loss 和 Alpha 进行了消融实验。

## 使用

1. 创建虚拟环境（可选），进入项目目录，安装所需包。

   ```bash
   cd TransFG
   pip install -r requirements.txt
   ```

2. 下载 [预训练模型](https://console.cloud.google.com/storage/vit_models/): ViT-B_16, ViT-B_32 并将其放置在 `pretrained_dir` 目录下，以便后续使用。可以通过命令如下来下载。

   ```bash
   wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
   ```

3. 下载相应数据集并放置在 `dataset_dir` 目录下以便后续使用。本项目复现所用数据集在介绍中有提供下载链接。

4. 执行命令进行训练，如果采用 CUB-200-2011 数据集进行训练，则 `dataset_name` 为 `CUB_200_2011` ，否则如果为 Stanford Dogs，则为 `dog` 。

   ```bash
   python3 train.py --dataset [dataset_name] --data_root [dataset_dir] --split [overlap/non-overlap] --pretrained_dir [pretrained_dir] --name [name]
   ```

## 模型性能

### Baseline

| 数据集与框架  | 准确率 |
| ------------- | ------ |
| CUB + Torch   | 91.7%  |
| Dogs + Torch  | 92.3%  |
| CUB + Jittor  | 91.0%  |
| Dogs + Jittor | 89.0%  |

### Patch Split

| Patch Split 与框架   | 准确率 | 训练时间 |
| -------------------- | ------ | -------- |
| Non-Overlap + Torch  | 91.5%  | 1.98h    |
| Overlap + Torch      | 91.7%  | 5.38h    |
| Non-overlap + Jittor | 90.2%  | 0.48h    |
| Overlap + Jittor     | 91.0%  | 1.00h    |

### Contrastive Loss

| 框架   | Contrastive Loss | 准确率 |
| ------ | ---------------- | ------ |
| Torch  | False            | 91.0%  |
| Torch  | True             | 91.5%  |
| Jittor | False            | 90.2%  |
| Jittor | True             | 91.0%  |

### Alpha

| 框架   | Alpha | 准确率 |
| ------ | ----- | ------ |
| Torch  | 0     | 91.1%  |
| Torch  | 0.2   | 91.4%  |
| Torch  | 0.4   | 91.7%  |
| Torch  | 0.6   | 91.5%  |
| Jittor | 0     | 90.4%  |
| Jittor | 0.2   | 90.5%  |
| Jittor | 0.4   | 91.0%  |
| Jittor | 0.6   | 90.7%  |

## 成员与分工

- 刘子恒：进行环境配置，数据集获取与导入，参与调试与实验。
- 徐烨堃：将 `torch` 有关内容替换为 `jittor` ，参与调试与实验。
- 殷尧瑞：进行性能分析，参与环境配置，调试与实验。

