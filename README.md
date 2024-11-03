[![arXiv](https://img.shields.io/badge/arXiv-2403.17633-b31b1b.svg)](https://arxiv.org/abs/2403.17633)

# UADA3D: Unsupervised Adversarial Domain Adaptation for 3D Object Detection with Sparse LiDAR and Large Domain Gaps

In this study, we address a gap in existing unsupervised domain adaptation approaches on LiDAR-based 3D object detection, which have predominantly concentrated on adapting between established, high-density autonomous driving datasets. We focus on sparser point clouds, capturing scenarios from different perspectives: not just from vehicles on the road but also from mobile robots on sidewalks, which encounter significantly different environmental conditions and sensor configurations. We introduce Unsupervised Adversarial Domain Adaptation for 3D Object Detection (\textbf{UADA3D}). UADA3D does not depend on pre-trained source models or teacher-student architectures. Instead, it uses an adversarial approach to directly learn domain-invariant features. We demonstrate its efficacy in various adaptation scenarios, showing significant improvements in both self-driving car and mobile robot domains.

![](https://maxiuw.github.io/uda/figures/main.png)



DATA AUGMENTATION:
Data augmentation is performed in both target and source domain. Make sure to only use world augmentations that do not alter object instances.

Data Preparation (Data Path given by folder name in data):
```
python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml ${DATA_PATH}
```

Test Model on LiDAR-CS-PUCK16:
```
python test.py --cfg_file cfgs/custom_models/puck-IA-SSD.yaml --batch_size 1 --ckpt ../output/custom_models/puck-IA-SSD/default/ckpt/checkpoint_epoch_80.pth --set MODEL.POST_PROCESSING.RECALL_MODE 'speed'
```

Test and Visualize model:
```
python vis.py --cfg_file ${CFG_FILE} --ckpt ${CKPT} --data_path ${POINT_CLOUD_DATA}
```


Train 64VLD Model with a batch size of 24:
```
python train.py --cfg_file ${CFG_FILE} --batch_size 24
```

DA testing command:
```
python adaptive_train.py --cfg_file cfgs/custom_models/${CFG_FILE} --batch_size 2
```

### Models
Pretrained models can be found [Here](https://kth-my.sharepoint.com/:f:/g/personal/maciejw_ug_kth_se/EqGk-27mGU5JkJOp1sm2bIABHrBS27HZMssBV61phyACrw?e=m3UNRX). 

Cite our work:
```bibtex
@article{wozniak2024uada3d,
  title={UADA3D: Unsupervised Adversarial Domain Adaptation for 3D Object Detection with Sparse LiDAR and Large Domain Gaps},
  author={Wozniak, Maciej K and Hansson, Mattias and Thiel, Marko and Jensfelt, Patric},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
