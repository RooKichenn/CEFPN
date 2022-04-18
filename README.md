# CEFPN论文代码复现
## 该项目主要使用的训练代码来自b站up主 霹雳吧啦wz：https://b23.tv/HvMiDy ，CEFPN论文代码纯手撸，如果转载，请标明出处。
# 环境配置：
①Python3.6/3.7/3.8

②Pytorch1.7.1(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)

③pycocotools(Linux:pip install pycocotools; Windows:pip install pycocotools-windows(不需要额外安装vs))

④Ubuntu或Centos(不建议Windows)

⑤最好使用GPU训练

⑥详细环境配置见requirements.txt

# 文件结构
    ├── backbone: 特征提取网络，可以根据自己的要求选择
  
    ├── network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
  
    ├── train_utils: 训练验证相关模块（包括cocotools）
  
    ├── my_dataset.py: 自定义dataset用于读取COCO数据集
  
    ├── train_mobilenet.py: 以MobileNetV2做为backbone进行训练
  
    ├── train_resnet50_fpn.py: 以resnet50+CEFPN做为backbone进行训练
  
    ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  
    ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  
    ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  
    └── pascal_voc_classes.json: pascal_voc标签文件
 # 预训练权重下载地址 
   ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth ，注意，下载的预训练权重记得要重命名，比如在train_resnet50_fpn.py中读取的是fasterrcnn_resnet50_fpn_coco.pth文件， 不是fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
  
# 数据集下载（默认使用的是COCO格式的数据集）
- COCO官网地址：https://cocodataset.org/
  
- 对数据集不了解的可以参考b站up主霹雳吧啦wz的博文：https://blog.csdn.net/qq_37541097/article/details/113247318
- 这里以下载coco2017数据集为例，主要下载三个文件：

    - 2017 Train images [118K/18GB]：训练过程中使用到的所有图像文件

    - 2017 Val images [5K/1GB]：验证过程中使用到的所有图像文件

    - 2017 Train/Val annotations [241MB]：对应训练集和验证集的标注json文件

都解压到coco2017文件夹下，可得到如下文件结构：

     ├── coco2017: 数据集根目录

     ├── train2017: 所有训练图像文件夹(118287张)
     
     ├── val2017: 所有验证图像文件夹(5000张)
     
     └── annotations: 对应标注文件夹
     
              ├── instances_train2017.json: 对应目标检测、分割任务的训练集标注文件
              
              ├── instances_val2017.json: 对应目标检测、分割任务的验证集标注文件
              
              ├── captions_train2017.json: 对应图像描述的训练集标注文件
              
              ├── captions_val2017.json: 对应图像描述的验证集标注文件
              
              ├── person_keypoints_train2017.json: 对应人体关键点检测的训练集标注文件
              
              └── person_keypoints_val2017.json: 对应人体关键点检测的验证集标注文件夹
# 训练方法
- 确保提前准备好数据集
- 确保提前下载好对应预训练模型权重
- 若要使用单GPU训练直接使用train_res50_fpn.py训练脚本
- 若要使用多GPU训练，使用torchrun --nproc_per_node=8 train_multi_GPU.py指令,nproc_per_node参数为使用GPU数量
- 如果想指定使用哪些GPU设备可在指令前加上CUDA_VISIBLE_DEVICES=0,3(例如只使用设备中的第1块和第4块GPU设备)CUDA_VISIBLE_DEVICES=0,3 torchrun --nproc_per_node=2 train_multi_GPU.py
# 注意事项
- 在使用训练脚本时，注意要将--data-path设置为自己存放coco2017文件夹所在的根目录
- 在使用预测脚本时，要将weights_path设置为你自己生成的权重路径。
- 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改--num-classes、--data-path和--weights-path即可，其他代码尽量不要改动
