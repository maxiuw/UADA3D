import argparse
import glob
from pathlib import Path
import pickle
import _init_path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
    print('Using open3d_vis')
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False
    print('Using mayavi_vis')

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, CustomDataset, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils




class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=None,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    print(args.data_path)

    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    '''
    val_set = CustomDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )
    '''
    if cfg.get('TARGET_DATA_CONFIG', None) is not None:
        dataset_cfg = cfg.TARGET_DATA_CONFIG
    else:
        dataset_cfg = cfg.DATA_CONFIG


    val_set, val_loader, sampler = build_dataloader(
        dataset_cfg=dataset_cfg,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        logger=logger,
        training=False,
    )

    logger.info(f'Total number of samples: \t{len(val_set)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=val_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(val_set):
            #idx = 310
            #data_dict = val_set[idx]
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = val_set.collate_batch([data_dict])

            print('frame_id = ' + str(data_dict['frame_id']))

            load_data_to_gpu(data_dict)
            pred_dicts, _, batch_dict = model.forward(data_dict)

            info_path = dataset_cfg.DATA_PATH + '/custom_infos_val.pkl'
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)

            # ANNOTATIONS
            info = infos[idx]
            annos = info['annos']
            gt_names = annos['name']

            # GROUND TRUTHS
            pcd = data_dict['points'][:, 1:]
            print('Point cloud size = ' + str(pcd.shape))
            gt_boxes = annos['gt_boxes_lidar']
            gt_scores = np.ones(gt_names.shape).astype(int)*1
            gt_labels = np.ones(gt_scores.shape).astype(int)*0

            # DETECTIONS
            pred_boxes = pred_dicts[0]['pred_boxes'].cpu()
            pred_scores = pred_dicts[0]['pred_scores'].cpu()
            pred_labels = np.ones(pred_scores.shape).astype(int)*1

            # CONCAT
            boxes = np.concatenate((gt_boxes, pred_boxes))
            labels = np.concatenate((gt_labels, pred_labels))
            scores = np.concatenate((gt_scores, pred_scores))
            print(pcd.shape)
            visualize_downsampling = False
            if visualize_downsampling:
                for sampled_points in batch_dict['encoder_xyz']:
                    sampled_points = sampled_points.cpu()[0,:,:]
                    print(sampled_points.shape)
                    point_colors = np.ones((sampled_points.shape[0], 3))
                    V.draw_scenes(
                        points=sampled_points, ref_boxes=boxes,
                        ref_scores=scores, ref_labels=labels,
                        point_colors = point_colors
                    )

                    if not OPEN3D_FLAG:
                        mlab.show(stop=True)

            V.draw_scenes(
                points=pcd, ref_boxes=boxes,
                ref_scores=scores, ref_labels=labels
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
