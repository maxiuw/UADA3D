from .detector3d_template import Detector3DTemplate

class DAIASSD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, batch_dict

    def get_training_loss(self, batch_dict=None):
        disp_dict = {}
        if batch_dict is not None:
            if batch_dict['use_det_loss']: # Always use detection loss in source domain 
                loss_point, tb_dict = self.point_head.get_loss()
            else:
                loss_point = 0
                tb_dict = None

        tb_dict = {} if tb_dict is None else tb_dict

        loss_discriminator, tb_dict = self.discriminator.get_loss(tb_dict)

        loss = loss_point + loss_discriminator

        return loss, tb_dict, disp_dict
