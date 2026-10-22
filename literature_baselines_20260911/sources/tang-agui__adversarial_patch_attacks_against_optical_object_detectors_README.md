本文件用于针对YOLO检测器训练对抗补丁，对应paper："Adversarial Patch Attacks against Aerial Imagery Object Detectors"
对应本人博士大论文第二章。
This project is an official implementation of the paper "Adversarial Patch Attacks against Aerial Imagery Object Detectors", which has been published in Neurocomputing.
- testfile
首先，运行，clean_img_pre.py，修改其中的imgdir，cfgfile和weightfile，以及savefile，
验证当前的权重文件（weightfile）和配置文件（cfgfile）是否可用。
python clean_img_pre.py
如果能正确输出程序中的内容，及说明此时设置的配置参数、环境等符合要求，可进行下一步训练。

- patch training file
训练补丁的主程序。
    - 参数设置：需要重点这是patch_config.py中的包括：
        - self.cfgfile：解析模型架构的参数，默认为"cfg/yolov3-dota.cfg"，即YOLOv3；
        - self.weightfile：YOLOv3的权重文件，默认为"weights/yolov3-dota_110000.weights"。该文件需要结合具体任务，例如DOTA数据集，VisDrone数据集，训练得到。
        - self.printfile：默认即可。
        - self.patch_size：补丁大小，默认为224
    - 数据集设置：在patch_config.py中的第14-17行（self.img_dir:图片，self.lab_dir：标签格式，本项目使用的标签格式为yolo格式，具体可参考相应数据集下格式要求）
    - 补丁保存设置：在主程序train_patch.py第309行，设置"save_patch_dir= training_patches_saving/training_test0809"
    补丁保存频率在314行设置。
训练补丁程序耗时较长，建议在后台训练，并保留日志：
nohup python -u train_patch.py > out_training/training_test/training_test_0809.out 2>&1 &
保留日志的路径可自己设置，当前设置路径out_training/training_test/training_test_0809.out仅用于演示。

- 其它注意事项data/：
    - dota.names：和数据集类别对应，用于作图；
    - simhei.ff：字体；
    - yolov3_anchors.txt：DOTA数据集的anchors。


**************************************************************8
本project已经在服务器上完成验证，具体使用的是刘旭博士账号下的/home/ubuntu/anaconda3/envs/patchAT虚拟环境。
如若有问题，可直接基于该虚拟环境，切换至/mnt/jfs/tangguijian/DOTA_YOLOv3_patch_AT/工程下即可。

运行本项目，必须有相应的权重文件和数据集，均可在服务器上找到。

Running epoch 0:   1%|          | 1/101 [00:25<42:07, 25.28s/it]
Running epoch 0:   2%|▏         | 2/101 [00:26<18:15, 11.06s/it]
输出log中出现以上内容表明程序已正常运行。
