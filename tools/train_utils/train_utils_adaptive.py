import glob
import os

import torch
import tqdm
import time
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils

def few_shot_fine_tune(model, optimizer, source_loader, target_loader, model_func, grl_scheduler, accumulated_iter,
                    optim_cfg, rank, tbar, total_it_each_epoch, source_dataloader_iter, target_dataloader_iter,
                    tb_log=None, leave_pbar=False,timers=None):
    if total_it_each_epoch == len(source_loader):
        source_dataloader_iter = iter(source_loader)

 
    pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
    data_time = common_utils.AverageMeter()
    batch_time = common_utils.AverageMeter()
    forward_time = common_utils.AverageMeter()
    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            source_batch = next(source_dataloader_iter)
        except StopIteration:
            source_dataloader_iter = iter(source_loader)
            target_dataloader_iter = iter(target_loader)
            source_batch = next(source_dataloader_iter)
            target_batch = next(target_dataloader_iter)
            print('new iters')
        try:
            target_batch = next(target_dataloader_iter)
        except:
            target_dataloader_iter = iter(target_loader)
            target_batch = next(target_dataloader_iter)

        data_timer = time.time()
        cur_data_time = data_timer - end

        grl_coeff = grl_scheduler.step(accumulated_iter)

        if cur_it < total_it_each_epoch // 2:
            cur_lr = 0.0001
        else:
            cur_lr = 0.00001

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
            tb_log.add_scalar('meta_data/gradient_reversal_coeff', grl_coeff, accumulated_iter)

        model.train()
        optimizer.zero_grad()


        source_batch['use_det_loss'] = True
        target_batch['use_det_loss'] = optim_cfg.get('SUPERVISED_ADAPTATION', False)

        source_batch['domain'] = 0
        target_batch['domain'] = 1

        source_batch['grl_coeff'] = grl_coeff
        target_batch['grl_coeff'] = grl_coeff

        source_loss, tb_dict, disp_dict = model_func(model, source_batch)
        target_loss, tb_dict_1, disp_dict_1 = model_func(model, target_batch)
        tb_dict.update(tb_dict_1)
        disp_dict.update(disp_dict_1)

        loss = source_loss + target_loss
        print(f"source_loss: {source_loss}, target_loss: {target_loss}")

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        data_time.update(avg_data_time)
        forward_time.update(avg_forward_time)
        batch_time.update(avg_batch_time)
        disp_dict.update({
            'loss': loss.item(), 'lr': cur_lr, 'grl': grl_coeff, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
            'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
        })

        pbar.update()
        pbar.set_postfix(dict(total_it=accumulated_iter))
        tbar.set_postfix(disp_dict)
        tbar.refresh()

        if tb_log is not None:
            tb_log.add_scalar('train/loss', loss, accumulated_iter)
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
            for key, val in tb_dict.items():
                tb_log.add_scalar('train/' + key, val, accumulated_iter)
    print(rank)
    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_one_epoch(model, optimizer, source_loader, target_loader, model_func, lr_scheduler, grl_scheduler, accumulated_iter,
                    optim_cfg, rank, tbar, total_it_each_epoch, source_dataloader_iter, target_dataloader_iter,
                    tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(source_loader):
        source_dataloader_iter = iter(source_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        
    for cur_it in range(total_it_each_epoch):
        end = time.time()
        try:
            source_batch = next(source_dataloader_iter)
        except StopIteration:
            source_dataloader_iter = iter(source_loader)
            target_dataloader_iter = iter(target_loader)
            source_batch = next(source_dataloader_iter)
            target_batch = next(target_dataloader_iter)
            print('new iters')
        try:
            target_batch = next(target_dataloader_iter)
        except:
            target_dataloader_iter = iter(target_loader)
            target_batch = next(target_dataloader_iter)

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)
        grl_coeff = grl_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
            tb_log.add_scalar('meta_data/gradient_reversal_coeff', grl_coeff, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        source_batch['use_det_loss'] = True
        target_batch['use_det_loss'] = optim_cfg.get('SUPERVISED_ADAPTATION', False)

        source_batch['domain'] = 0
        target_batch['domain'] = 1

        source_batch['grl_coeff'] = grl_coeff
        target_batch['grl_coeff'] = grl_coeff

        source_loss, tb_dict, disp_dict = model_func(model, source_batch)
        target_loss, tb_dict_1, disp_dict_1 = model_func(model, target_batch)
        tb_dict.update(tb_dict_1)
        disp_dict.update(disp_dict_1)

        loss = source_loss + target_loss

        forward_timer = time.time()
        cur_forward_time = forward_timer - data_timer

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1

        cur_batch_time = time.time() - end
        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'grl': grl_coeff, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })

            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, source_loader, target_loader, model_func, lr_scheduler, grl_scheduler,
                optim_cfg, start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                source_sampler=None, target_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(source_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(source_loader.dataset, 'merge_all_iters_to_one_epoch')
            source_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(source_loader) // max(total_epochs, 1)

        source_dataloader_iter = iter(source_loader)
        target_dataloader_iter = iter(target_loader)
        for cur_epoch in tbar:
            if source_sampler is not None and target_sampler is not None:
                source_sampler.set_epoch(cur_epoch)
                target_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, source_loader, target_loader, model_func,
                lr_scheduler=cur_scheduler, grl_scheduler=grl_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                source_dataloader_iter=source_dataloader_iter,
                target_dataloader_iter=target_dataloader_iter
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )
    # FEW-SHOT FINE-TUNING
    if optim_cfg.get('FEW_SHOT_FINETUNING', False):
        print('**********************Start few-shot finetuning**********************')

        source_dataloader_iter = iter(source_loader)
        target_dataloader_iter = iter(target_loader)

        accumulated_iter = few_shot_fine_tune(
            model, optimizer, source_loader, target_loader, model_func,
            accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
            rank=rank, tbar=tbar, tb_log=tb_log, grl_scheduler=grl_scheduler,
            total_it_each_epoch=total_it_each_epoch,
            source_dataloader_iter=source_dataloader_iter,
            target_dataloader_iter=target_dataloader_iter
        )

        trained_epoch = optim_cfg.NUM_EPOCHS + 1

        ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
        ckpt_list.sort(key=os.path.getmtime)

        ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
        save_checkpoint(
            checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
        )



def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
