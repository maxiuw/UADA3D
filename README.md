
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
python train.py --cfg_file cfgs/custom_models/VLD64-IA-SSD.yaml --batch_size 24
```

DA testing command:
```
python adaptive_train.py --cfg_file cfgs/custom_models/da-testing-IA-SSD.yaml --batch_size 2
```

### Models
Pretrained models can be found [Here](https://kth-my.sharepoint.com/:f:/g/personal/maciejw_ug_kth_se/EqGk-27mGU5JkJOp1sm2bIABHrBS27HZMssBV61phyACrw?e=m3UNRX). 


