from motion_vae.config import *
from motion_vae.dataset import Video3DPoseDataset
from motion_vae.model import PoseMixtureVAE
from utils.konia_transform import quaternion_to_angle_axis
from utils.torch_transform import rot6d_to_angle_axis

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import normalize
import time
from collections import defaultdict
import logging
import copy


###############################################################################
# Train pipeline
###############################################################################

class MotionVAEModel(object):

    def __init__(self, opt, device=None):
        self.opt = opt
        if device is not None:
            self.device = device
        else:
            if opt.gpu_ids:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        frame_size_condition = frame_size_truth = frame_size_pred = opt.frame_size
        if opt.predict_phase:
            frame_size_pred += 2
        self.model = PoseMixtureVAE(
            frame_size_condition,
            frame_size_truth,
            frame_size_pred,
            opt.latent_size,
            opt.hidden_size,
            opt.num_condition_frames,
            opt.num_future_predictions,
            opt.num_experts,
        )
        self.model.to(self.device)
        if opt.test_only: 
            self.model.eval()

        self.checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.model_ver)
        if not opt.test_only and not opt.continue_train:
            if os.path.exists(self.checkpoint_dir):
                logging.warning("Checkpoint already exists!")
                # raise Exception("Checkpoint already exists!")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            logging.info('Checkpoint saved into {}'.format(self.checkpoint_dir))
      
        if opt.test_only: 
            self.load_checkpoint()
            return 

        if not opt.no_log:
            self.writer = SummaryWriter(os.path.join(self.checkpoint_dir, 'logs'))

        if opt.continue_train:
            self.load_checkpoint()
            logging.info("Continue training with latest model")
        
        self.dataset = Video3DPoseDataset(opt)
        self.dataset.get_normalization_stats()
  
        self.trainset = DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=int(opt.num_threads))
        
        if self.opt.mixed_phase_schedule:
            opt_no_phase = copy.deepcopy(opt)
            opt_no_phase.predict_phase = False
            self.dataset_no_phase = Video3DPoseDataset(opt_no_phase)
            self.dataset_no_phase.set_normalization_stats(self.dataset.avg, self.dataset.std)

            self.trainset_no_phase = DataLoader(
                self.dataset_no_phase,
                batch_size=opt.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=int(opt.num_threads))

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        self.old_lr = opt.lr

        if opt.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        if opt.softmax_future:
            self.future_weights = torch.softmax(
                torch.linspace(1, 0, opt.num_future_predictions), dim=0
            ).to(self.device)
        else:
            self.future_weights = (
                torch.ones(opt.num_future_predictions)
                .to(self.device)
                .div_(opt.num_future_predictions)
            ).to(self.device)

    #-------------------------------------------------- Training ------------------------------------------------------------

    def train(self):
        opt = self.opt
        total_iters = 0
        train_start_time = time.time()
        train_loss_dict = defaultdict(float)
        num_batch_phase = 0

        for epoch in range(0, opt.n_epochs + opt.n_epochs_decay):
            epoch_start_time = time.time()
            if opt.mixed_phase_schedule:
                trainset_no_phase_sampler = iter(self.trainset_no_phase)

            for _, batch_data in enumerate(self.trainset):
                if opt.mixed_phase_schedule:
                    batch_data_no_phase = next(trainset_no_phase_sampler)
                    sample_phase = self.schedual_mixed_phase(epoch)
                    if not sample_phase:
                        batch_data = batch_data_no_phase

                total_iters += opt.batch_size
                regressive = self.schedual_regressive_training(epoch)
                loss_dict = self.train_batch(batch_data, regressive)
                num_batch_phase += batch_data.get('phase') is not None

                for k, v in loss_dict.items():
                    train_loss_dict[k] += v

                if total_iters % opt.log_freq == 0:
                    for k, v in train_loss_dict.items():
                        if k != 'recon_phase':
                            train_loss_dict[k] = v / (opt.log_freq / opt.batch_size)
                        else:
                            train_loss_dict[k] = v / num_batch_phase

                    # dump loss to log
                    if not opt.no_log:
                        for k, v in train_loss_dict.items():
                            self.writer.add_scalar('train_loss/{}'.format(k), v, total_iters)
                    # print loss to console
                    self.print_loss(epoch + 1, total_iters, time.time() - train_start_time, train_loss_dict)
                    train_loss_dict = defaultdict(float)
                    num_batch_phase = 0

            # save latest model
            if not opt.no_log:
                self.save_checkpoint(label='latest')
                # save checkpoint
                if (epoch + 1) % opt.save_freq_epoch == 0:
                    logging.info('Saving the latest model (epoch %d, total_iters %d)\n' % (epoch + 1, total_iters))
                    self.save_checkpoint(label='epoch_{}'.format(epoch + 1))

            logging.info('End of epoch %d / %d \t Time Taken: %d sec\n' % (
                epoch + 1, opt.n_epochs + opt.n_epochs_decay, time.time() - train_start_time))

            if epoch > opt.n_epochs:
                self.update_learning_rate()
        if not opt.no_log:
            self.writer.close()
        logging.info('End of training Time Taken: %d sec\n' % (time.time() - train_start_time))


    def train_batch(self, batch_data, regressive=False):
        opt = self.opt
        B, L, F = batch_data['feature'].shape
        T = opt.num_condition_frames
        S = opt.num_future_predictions
        batch_feature = batch_data['feature'].float().to(self.device)
        batch_phase = None
        if batch_data.get('phase') is not None:
            batch_phase = batch_data['phase'].float().to(self.device)
        batch_action = None
        if batch_data.get('action') is not None:
            batch_action = batch_data['action'].float().to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        loss_dict_seq = defaultdict(float)

        condition = batch_feature[:, :T].clone()

        for i in range(T-1, L - S):
            # set input
            if i >= T:
                condition = condition.roll(-1, dims=1)
                condition[:, -1].copy_(output[:, 0].detach() if regressive else batch_feature[:, i])

            gt_feature = batch_feature[:, i+1:i+1+S]
            if batch_phase is not None:
                gt_phase = batch_phase[:, i+1:i+1+S]
            else:
                gt_phase = None
            if batch_action is not None:
                gt_action = batch_action[:, i+1:i+1+S]
            else:
                gt_action = None

            # forward pass
            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                (output, output_phase, _, _), loss_dict = self.feed_vae(gt_feature, condition, gt_phase, gt_action)

            # backward pass
            self.optimizer.zero_grad()
            loss_total = sum(loss_dict.values())
            if opt.use_amp:
                self.scaler.scale(loss_total).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_total.backward()
                self.optimizer.step()

            for k, v in loss_dict.items():
                loss_dict_seq[k] += v.item()

        for k, v in loss_dict_seq.items():
            loss_dict_seq[k] = v / ((L - S - T + 1))
        return loss_dict_seq
    

    def feed_vae(self, ground_truth, condition, gt_phase=None, gt_action=None):
        opt = self.opt
        condition = condition.flatten(start_dim=1, end_dim=2)
        flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2)
        if gt_action is not None:
            condition = torch.cat([condition, gt_action.flatten(start_dim=1, end_dim=2)], dim=1)

        output, mu, logvar = self.model(flattened_truth, condition)
        output_phase = None
        if not opt.predict_phase:
            output = output.view(-1, opt.num_future_predictions, opt.frame_size)
        else:
            output = output.view(-1, opt.num_future_predictions, opt.frame_size + 2)
            output_phase = output[:, :, -2:]
            output = output[:, :, :-2]

        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum().clamp(max=0)
        kl_loss /= logvar.numel()

        recon_feature_loss = (output - ground_truth.detach()).pow(2).mean(dim=(0, -1))
        recon_feature_loss = recon_feature_loss.mul(self.future_weights).sum()

        if opt.predict_phase and gt_phase is not None:
            recon_phase_loss = (output_phase - gt_phase.detach()).pow(2).mean(dim=(0, -1))
            recon_phase_loss = recon_phase_loss.mul(self.future_weights).sum()

        loss_dict = {
            'recon': recon_feature_loss * self.opt.weights['recon'],
            'kl': kl_loss * self.opt.weights['kl'],
        }
        if opt.predict_phase and gt_phase is not None:
            loss_dict['recon_phase'] = recon_phase_loss * self.opt.weights['recon_phase']
        return (output, output_phase, mu, logvar), loss_dict


    def save_checkpoint(self, label=None):
        state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(self.checkpoint_dir, '{}.tar'.format(label)))
        if self.opt.use_amp:
            torch.save(self.scaler.state_dict(), os.path.join(self.checkpoint_dir, '{}_scaler.tar'.format(label)))
        np.save(os.path.join(self.checkpoint_dir, 'avg.npy'), self.dataset.avg)
        np.save(os.path.join(self.checkpoint_dir, 'std.npy'), self.dataset.std)


    def load_checkpoint(self, label='latest'):
        model_state_dict_path = os.path.join(self.checkpoint_dir, '{}.tar'.format(label))
        self.model.load_state_dict(torch.load(model_state_dict_path))
        logging.info("MotionAE checkpoint loaded from {}!".format(model_state_dict_path))

        self.avg = torch.from_numpy(np.load(os.path.join(self.checkpoint_dir, 'avg.npy'))).float().to(self.device)
        self.std = torch.from_numpy(np.load(os.path.join(self.checkpoint_dir, 'std.npy'))).float().to(self.device)


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.n_epochs_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        logging.info('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


    def schedual_regressive_training(self, epoch):
        opt = self.opt
        if opt.curriculum_schedule is None:
            return True
        l = int((opt.n_epochs + opt.n_epochs_decay) * opt.curriculum_schedule[0])
        h = int((opt.n_epochs + opt.n_epochs_decay) * opt.curriculum_schedule[1])
        thresh = min(h-l, max(0, epoch - l)) / (h - l)
        return np.random.rand() <= thresh
    

    def schedual_mixed_phase(self, epoch):
        (e1, t1), (e2, t2) = self.opt.mixed_phase_schedule
        opt = self.opt
        l = int((opt.n_epochs + opt.n_epochs_decay) * e1)
        h = int((opt.n_epochs + opt.n_epochs_decay) * e2)
        p = min(h-l, max(0, epoch - l)) / (h - l)
        if p == 0:
            thresh = t1
        elif p == 1:
            thresh = t2
        else:
            thresh = t1 + (t2 - t1) * p
        return np.random.rand() <= thresh


    def print_loss(self, epoch, iter, t, loss_dict):
        message = '(epoch: %d, iters: %d, time: %d) ' % (epoch, iter, t)
        for k, v in loss_dict.items():
            if v != 0:
                message += '%s: %.5f ' % (k, v)
        logging.info(message)
    

    def normalize(self, tensor, inds=None):
        if inds is None:
            return (tensor - self.avg) / self.std
        else:
            return (tensor - self.avg[inds]) / self.std[inds]


    def unnormalize(self, tensor):
        return tensor * self.std + self.avg


    def infer_single(self, latent, condition, action=None):
        opt = self.opt
        assert latent.shape == (opt.latent_size,)
        assert condition.shape == (opt.num_condition_frames, opt.frame_size)
        frame = {}
        with torch.no_grad():
            z = latent.unsqueeze(0).to(self.device)
            c = self.normalize(condition.to(self.device)). \
                unsqueeze(0).flatten(start_dim=1, end_dim=2)
            output = self.model.sample(z, c)
            if not opt.predict_phase:
                output = output.view(opt.num_future_predictions, opt.frame_size)
            else:
                output = output.view(opt.num_future_predictions, opt.frame_size + 2)
                output_phase = output[:, -2:]
                output = output[:, :-2]
            output = self.unnormalize(output).detach().cpu()
            root_dim = 3 if 'root_pos' not in opt.pose_feature else 6
            frame = {
                'root_velo': output[0, root_dim-3 : root_dim],
                'feature': output[0]
            }
            if opt.update_joint_pos:
                frame['joint_pos'] = output[0, root_dim:root_dim+24*3]
            else:
                joint_rot_dim = root_dim
                if 'joint_pos' in opt.pose_feature:
                    joint_rot_dim += 23*3
                if 'joint_velo' in opt.pose_feature:
                    joint_rot_dim += 23*3
                
                if 'joint_rotmat' in opt.pose_feature:
                    # Need to switch from row major to column major
                    joint_rot6d = output[0, joint_rot_dim:joint_rot_dim+24*6].reshape(24, 6)
                    frame['joint_rot'] = rot6d_to_angle_axis(joint_rot6d)
                if 'joint_quat' in opt.pose_feature:
                    joint_quat = output[0, joint_rot_dim:joint_rot_dim+24*4].view(24, 4)
                    # TO CHECK: Do we need to normalize?
                    joint_quat = normalize(joint_quat, dim=-1)
                    frame['joint_rot'] = quaternion_to_angle_axis(joint_quat)
            if opt.predict_phase:
                phase = output_phase[0].detach().cpu().reshape(-1)
                phase_rad = torch.atan2(phase[0], phase[1]).numpy()
                if phase_rad < 0: phase_rad += np.pi * 2
                frame['phase'] = phase
                frame['phase_rad'] = phase_rad
                
        return frame


    def forward(self, action, condition):
        opt = self.opt
        z = action.float()
        c = condition.float() # normalized already

        # use amp to speed up inference
        with torch.cuda.amp.autocast():
            output = self.model.sample(z, c)
        output = output.float()

        output_phase = None
        if not opt.predict_phase:
            output = output.view(-1, opt.num_future_predictions, opt.frame_size)
        else:
            output = output.view(-1, opt.num_future_predictions, opt.frame_size + 2)
            output_phase = output[:, :, -2:]
            output = output[:, :, :-2]
        return output[:, 0], output_phase[:, 0]