# Diffusion Policy Policy Optimization (DPPO)

## 基础说明

本代码基于 [DPPO](https://github.com/irom-princeton/dppo) 官方代码库开发。已接入 homebot 仓库中的环境和对应的扩散策略网络。

## 环境配置

可以根据 dppo_sapien.yaml 配置环境，运行如下指令。

```bash
conda env create -f dppo_sapien.yaml
conda activate dppo_sapien
pip install -e .
```


## 训练准备

把 homebot 的 asset 文件夹软连接到当目录中，可以使用如下命令：

```bash
ln -s \path\to\asset asset
```

## 训练脚本

根据 `cfg\sapien\finetune` 中的参数进行 DPPO 训练。

训练需要提供预训练文件的路径（包含模型路径和正则化文件路径），以及提供 log 输出的地址。

例如，训练 `drawe_push` 环境，可以使用如下脚本:

```bash
python script/run.py --config-name=ft_ppo_diffusion_img --config-dir=cfg/sapien/finetune/microwave \
run_dir=\path\to\run_dir base_policy_path=\path\to\pretrained_model normalization_path=\path\to\normalization_status
```
