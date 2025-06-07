# Diffusion Policy Policy Optimization (DPPO)

## 基础说明
本代码基于 [DPPO](https://github.com/irom-princeton/dppo) 官方代码库开发。已接入 homebot 仓库中的环境和对应的扩散策略网络。

## 环境配置
使用dppo_sapien.yaml创建Conda环境，并 `pip intall -e.`。 
如需要maniskill egad数据集，需要去egad网站下载解压，并使用coacd计算碰撞（目前dppo脚本中暂时不用egad数据集）。

## 环境说明
涉及抓放物体、推拉抽屉、门窗、微波炉等环境。路径置于 ``env/homebot`` 之下。和 homebot 仓库中环境的区别在于加入了简单的奖励函数设计。
物体资产与 homebot 仓库一致。

## 策略说明
### 策略结构
使用 DPPO 算法。其中 Actor 网络与模仿学习获得的扩散策略一致(diffusion_unet文件下定义)，Critic 网络采用 DPPO 默认的基于 ViT 的网络实现。
### 策略训练
配置文件见 ``cfg/sapien``。运行脚本见 ``test.bash``。