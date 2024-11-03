import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss

from .gradient_reversal import GradientReversal

"""
    adversarial domain discriminator built from config
"""
class Discriminator2(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        # Read config
        self.model_cfg = model_cfg
        self.loss_cfg = model_cfg.LOSS_CONFIG

        self.marginal = False

        if self.model_cfg.get('DA_SETTINGS', None) is not None:
            self.marginal = True
            self.settings_cfg = model_cfg.DA_SETTINGS

            # Save for forward pass
            self.input_dict_key = self.settings_cfg.INPUT_DICT_KEY
            self.input_dict_layer = self.settings_cfg.INPUT_DICT_LAYER

            # Save for network initialization
            mlp_types = self.settings_cfg.MLP_TYPE
            mlps = self.settings_cfg.MLPS

            self.discriminators = nn.ModuleList()

            if self.loss_cfg.get('LOSS_FUNCTION')[0] == 'LeastSquares':
                use_sigmoid_M = True
            else:
                use_sigmoid_M = False

            for i in range(len(mlp_types)):
                kernel_size = self.settings_cfg.get('KERNEL_SIZE', 1)
                mlps[i].insert(0, self.settings_cfg.NUM_FEATURES)

                discriminator = self.build_mlps(True, mlps[i], mlp_types[i], kernel_size=kernel_size,
                                                use_sigmoid=use_sigmoid_M)

                self.discriminators.append(nn.Sequential(*discriminator))

        # Conditional probability distribution alignment, P(I|y,b)
        self.conditional = False

        if self.loss_cfg.get('LOSS_CONDITIONAL') == 'LeastSquares':
            use_sigmoid_C = True
        else:
            use_sigmoid_C = False

        if self.model_cfg.get('CONDITIONAL_ADAPTATION', None) is not None:
            self.conditional = True
            self.cond_cfg = model_cfg.CONDITIONAL_ADAPTATION
            cond_MLPS = self.cond_cfg.MLPS

            cond_MLPS.insert(0, self.cond_cfg.NUM_FEATURES +
                             self.cond_cfg.BOX_REGRESSION_PARAMS)

            self.cond_cls_key = self.cond_cfg.INPUT_DICT_KEYS[0]
            self.cond_box_key = self.cond_cfg.INPUT_DICT_KEYS[1]
            self.cond_feat_key = self.cond_cfg.INPUT_DICT_KEYS[2]
            kernel_size_C = self.cond_cfg.get('KERNEL_SIZE', 1)

            self.cond_discriminators = nn.ModuleList()

            for i in range( self.cond_cfg.NUM_CLASSES ):
                cond_discriminator = self.build_mlps(True, cond_MLPS, self.cond_cfg.MLP_TYPE,
                                                     kernel_size=kernel_size_C,
                                                     use_sigmoid=use_sigmoid_C)
                self.cond_discriminators.append(nn.Sequential(*cond_discriminator))

        self.regularization = self.loss_cfg.get('CONSISTENCY_REGULARIZATION', False)

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

    def forward(self, batch_dict):

        ret_dict = {}
        #print(batch_dict.keys())

        if self.training:
            if self.marginal:
                if batch_dict.get('grl_coeff', None) is not None:
                    for i in range(len(self.input_dict_layer)):
                        self.discriminators[i][0].update_lambda(batch_dict['grl_coeff'])

                domain_preds = []
                domain_refs = []
                for i in range(len(self.input_dict_layer)):

                    if self.input_dict_layer[i]:
                        domain_pred = self.discriminators[i](
                            batch_dict[self.input_dict_key[i]][self.input_dict_layer[i][0]]
                        )
                    else:
                        domain_pred = self.discriminators[i](
                            batch_dict[self.input_dict_key[i]]
                        )

                    domain_ref = domain_pred.detach().clone()
                    domain_ref[:] = batch_dict['domain']

                    domain_preds.append(domain_pred)
                    domain_refs.append(domain_ref)


                ret_dict['domain_preds'] = domain_preds
                ret_dict['domain_refs'] = domain_refs

            if self.conditional:
                cond_preds = []
                cond_refs = []

                for i in range( len(self.cond_discriminators) ):
                    if self.cond_cfg.MLP_TYPE == 'Conv2d':
                        cls_feats = torch.cat([batch_dict[self.cond_feat_key],
                                               batch_dict[self.cond_box_key]], 1)
                        y_k = batch_dict[self.cond_cls_key][:,i:i+1,:,:]
                        input_tensor = y_k*cls_feats
                    else:
                        cls_feats = torch.cat([batch_dict[self.cond_feat_key],
                                               batch_dict[self.cond_box_key]], 1)
                        cls_feats = torch.transpose(cls_feats, 0, 1)
                        input_tensor = batch_dict[self.cond_cls_key][:,i]*cls_feats
                        input_tensor = torch.transpose(input_tensor, 0, 1)


                    cond_pred = self.cond_discriminators[i](input_tensor)

                    cond_ref = cond_pred.detach().clone()
                    cond_ref[:] = batch_dict['domain']

                    cond_preds.append(cond_pred)
                    cond_refs.append(cond_ref)

                ret_dict['cond_preds'] = cond_preds
                ret_dict['cond_refs'] = cond_refs

        self.forward_ret_dict = ret_dict

        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict


        loss_refs_dict = {
            'CrossEntropy': binary_cross_entropy_with_logits,
            'LeastSquares': mse_loss
        }

        disc_loss_total = 0

        if self.marginal:
            domain_preds = self.forward_ret_dict['domain_preds']
            domain_refs =  self.forward_ret_dict['domain_refs']

            for i in range(len(self.loss_cfg.LOSS_FUNCTION)):
                __loss_func__ = loss_refs_dict[self.loss_cfg.LOSS_FUNCTION[i]]
                disc_loss = __loss_func__(domain_preds[i], domain_refs[i])

                tb_dict.update({f"disc_loss{i}": disc_loss.item()})
                disc_loss_total = disc_loss_total + disc_loss

        if self.conditional:
            cond_preds = self.forward_ret_dict['cond_preds']
            cond_refs = self.forward_ret_dict['cond_refs']
            reg_loss = 0
            __loss_func__ = loss_refs_dict[self.loss_cfg.LOSS_CONDITIONAL]
            for i in range(len(cond_preds)):
                cond_loss = __loss_func__(cond_preds[i], cond_refs[i])
                tb_dict.update({f"cond_disc_loss{i}": cond_loss.item()})
                disc_loss_total = disc_loss_total + cond_loss

                if self.regularization:
                    reg_loss = reg_loss + __loss_func__(cond_preds[i], domain_preds[0])

        if self.regularization:
            tb_dict.update({f"reg_loss{i}": reg_loss.item()})
            disc_loss_total = disc_loss_total + reg_loss

        return disc_loss_total, tb_dict


    def get__discriminator_regularization_loss(self, tb_dict=None):
        raise NotImplementedError

    def build_mlps(self, use_grl,mlps, mlp_type, kernel_size=1, use_sigmoid=False):
        discriminator = []
        if use_grl:
            discriminator.extend([ GradientReversal() ])

        if mlp_type == 'Conv1d':

            for k in range(len(mlps) - 1):
                discriminator.extend([
                    nn.Conv1d(mlps[k], mlps[k + 1], kernel_size=kernel_size, bias=False),
                    nn.LeakyReLU()
                ])
            discriminator.extend([ nn.Conv1d(mlps[-1], 1, kernel_size=kernel_size, bias=True) ])

        elif mlp_type == 'Linear':

            for k in range(len(mlps) - 1):
                discriminator.extend([
                    nn.Linear(mlps[k], mlps[k + 1], bias=False),
                    nn.LeakyReLU()
                ])
            discriminator.extend([ nn.Linear(mlps[-1], 1, bias=True) ])

        elif mlp_type == 'Conv2d':

            for k in range(len(mlps) - 1):
                discriminator.extend([
                    nn.Conv2d(mlps[k], mlps[k + 1], kernel_size=kernel_size, bias=False),
                    nn.LeakyReLU()
                ])
            discriminator.extend([ nn.Conv2d(mlps[-1], 1, kernel_size=kernel_size, bias=True) ])

        else:
            raise NotImplementedError

        if use_sigmoid:
            discriminator.extend([ nn.Sigmoid() ])

        return discriminator


class Discriminator(nn.Module):
    def __init__(self, model_cfg, num_features):
        super().__init__()
        self.model_cfg = model_cfg
        # TODO: CHANGE HOW INPUT POINTS ARE READ FROM CONFIG

        self.mlp = nn.Sequential(
            GradientReversal(),
            nn.Linear(num_features, 256, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 1, bias=True)
        )

        self.ctr_mlp = nn.Sequential(
            GradientReversal(),
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(64, 32, kernel_size=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=1, bias=True)
        )

        #self.cent_mlp2 = nn.Sequential(
        #    GradientReversal(alpha=0.1),
        #)

    def forward(self, batch_dict):
        """
        input x: center features (N, C)
        output: feature-wise domain probability (N) in (0,1)
        """

        if batch_dict.get('grl_coeff', None) is not None:
            self.mlp[0].update_lambda(batch_dict['grl_coeff'])
            self.ctr_mlp[0].update_lambda(batch_dict['grl_coeff'])

        domain_preds = self.mlp(batch_dict['centers_features'])
        domain_ctr_preds = self.ctr_mlp(batch_dict['encoder_features'][3])

        domain_refs = domain_preds.detach().clone()
        domain_refs[:] = batch_dict['domain']

        domain_ctr_refs = domain_ctr_preds.detach().clone()
        domain_ctr_refs[:] = batch_dict['domain']

        ret_dict = {
            'domain_preds': domain_preds,
            'domain_refs': domain_refs,
            'domain_ctr_preds': domain_ctr_preds,
            'domain_ctr_refs': domain_ctr_refs
        }

        self.forward_ret_dict = ret_dict

        print(batch_dict['points_features'])

        return batch_dict

    def get_loss(self, tb_dict=None):
        """
        config references:
        LOSS_GLOBAL: CrossEntropy
        LOSS_LOCAL: None
        LOSS_REG: None
        """
        tb_dict = {} if tb_dict is None else tb_dict

        if self.model_cfg.LOSS_CONFIG.get('LOSS_GLOBAL', None) is not None:
            disc_loss_global, tb_dict_0 = self.get_global_discriminator_loss()
            tb_dict.update(tb_dict_0)
        else:
            disc_loss_global = 0

        if self.model_cfg.LOSS_CONFIG.get('LOSS_CTR', None) is not None:
            disc_loss_ctr, tb_dict_1 = self.get_ctr_discriminator_loss()
            tb_dict.update(tb_dict_1)
        else:
            disc_loss_ctr = 0

        if self.model_cfg.LOSS_CONFIG.get('LOSS_REG', None) is not None:
            disc_loss_reg, tb_dict_2 = self.get_discriminator_regularization_loss()
            tb_dict.update(tb_dict_2)
        else:
            disc_loss_reg = 0

        disc_loss = disc_loss_global + disc_loss_ctr

        return disc_loss, tb_dict

    def get_global_discriminator_loss(self, tb_dict=None):
        disc_loss_global = 0

        domain_preds = self.forward_ret_dict['domain_preds']
        domain_refs =  self.forward_ret_dict['domain_refs']

        loss_refs_dict = {
            'CrossEntropy': binary_cross_entropy_with_logits,
            'LeastSquares': mse_loss
        }

        __loss_func__ = loss_refs_dict[self.model_cfg.LOSS_CONFIG.get('LOSS_GLOBAL')]
        disc_loss_global = __loss_func__(domain_preds, domain_refs)

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'disc_loss_global': disc_loss_global})

        return disc_loss_global, tb_dict

    def get_ctr_discriminator_loss(self, tb_dict=None):
        domain_ctr_preds = self.forward_ret_dict['domain_ctr_preds']
        domain_ctr_refs = self.forward_ret_dict['domain_ctr_refs']

        loss_refs_dict = {
            'CrossEntropy': binary_cross_entropy_with_logits,
            'LeastSquares': mse_loss
        }

        __loss_func__ = loss_refs_dict[self.model_cfg.LOSS_CONFIG.get('LOSS_CTR')]
        disc_loss_ctr = __loss_func__(domain_ctr_preds, domain_ctr_refs)

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'disc_loss_ctr': disc_loss_ctr})

        return disc_loss_ctr, tb_dict

    def get__discriminator_regularization_loss(self, tb_dict=None):
        raise NotImplementedError
