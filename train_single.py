import os
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.timer import Timer
from utils.logging import get_logger

from models.vithand import getJointTokenModel, load_vitpose_b
from models.optimizer import get_optimizer_grouped_params
from datasets.dexycb import DexYCB_Temporal
from engine import evaluate_joints

import configs.sar_dexycb_64x64 as vitpose_cfg
from config import cfg


class processer(object):
    def __init__(self, args):
        self.args = args
        self.device = cfg.device
        output_dir = os.path.join(self.args.output_root, self.args.experiment, 'train')
        log_folder = os.path.join(output_dir, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, self.args.experiment + '.log')
        self.logger = get_logger(log_file=logfile, name="Training")
        self.logger.info(f'Start training: %s' % ('train_' + self.args.experiment))
        self.writer = SummaryWriter(os.path.join(output_dir,'run'))
        self.save_path = os.path.join(output_dir,'checkpoints')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.timer = Timer()

        self.early_stop = 0
        self.min_joint_err = 1e5

    def make_model(self):
        self.start_epoch = 0
        self.model = getJointTokenModel().to(self.device)
        self.optimizer = self.get_layerdecay_optimizer(self.model)
        self.scheduler = self.get_scheduler(self.optimizer)

        if self.args.checkpoint:
            result = load_vitpose_b(self.model, self.args.checkpoint)
            self.logger.info(f'Load ViTPose-B weight from checkpoint:{self.args.checkpoint}')
            self.logger.info('[missing_keys: {}]'.format(len(result.missing_keys)))
            self.logger.info('[unexpected_keys: {}]'.format(len(result.unexpected_keys)))

        if self.args.continue_train:
            ckpt_path = os.path.join(self.args.output_root, self.args.experiment, 'train/checkpoints/ckpt_last.pth')
            ckpt = torch.load(ckpt_path, map_location='cpu')
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['network'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.last_epoch = self.start_epoch
            self.logger.info(f'Continue from ckeckpoint:{ckpt_path}')

        self.logger.info("The model ViTHand is made successfully.")

    def load_data(self):
        self.train_set = DexYCB_Temporal(self.args.data_root, 'train', T=0, gap=self.args.seq_gap)
        self.val_set = DexYCB_Temporal(self.args.data_root, 'test', T=0, gap=self.args.seq_gap)
        self.train_loader = DataLoader(self.train_set,
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       num_workers=self.args.num_workers,
                                       pin_memory=True,
                                       drop_last=True)
        self.val_loader = DataLoader(self.val_set,
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=self.args.num_workers,
                                     pin_memory=True,
                                     drop_last=False)

    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epoch, eta_min=1e-5)

    def get_layerdecay_optimizer(self, model):
        # Layer-wise learning rate decay
        param_groups = get_optimizer_grouped_params(model,
                                                    vitpose_cfg.optimizer['paramwise_cfg']['layer_decay_rate'],
                                                    vitpose_cfg.optimizer['lr'],
                                                    vitpose_cfg.optimizer['weight_decay'],
                                                    vitpose_cfg.optimizer['paramwise_cfg']['num_layers'])
        optimizer = optim.AdamW(param_groups.values(), vitpose_cfg.optimizer['lr'])

        return optimizer

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def save_model(self, state, epoch):
        file_path = os.path.join(self.save_path, 'ckpt_{}.pth'.format(epoch))
        torch.save(state, file_path)
        self.logger.info("Save checkpoint into {}".format(file_path))

    def start_train(self):
        for epoch in range(self.start_epoch, self.args.epoch):
            self.logger.info('Start training epoch:{}'.format(epoch+1))
            self.train(epoch)
            self.logger.info('Start validation epoch:{}'.format(epoch+1))
            self.test(epoch)
            if self.early_stop >= self.args.early_stop:
                break

    def train(self, epoch):
        self.model.train()
        timer = {'dataloader': 0., 'model': 0., 'statistics': 0.}
        self.timer.tic()
        for itr, (inputs, targets, meta_infos) in enumerate(self.train_loader):
            timer['dataloader'] += self.timer.toc()

            img = inputs['img'][:,-1].to(self.device)
            joints = targets['guide_pose'][:,-1].to(self.device)
            # forward
            self.optimizer.zero_grad()
            _, pos = self.model(img)
            loss = F.smooth_l1_loss(pos, joints[:,:,:2], beta=0.03)

            timer['model'] += self.timer.toc()

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), **vitpose_cfg.optimizer_config['grad_clip'])
            self.optimizer.step()

            # visualize
            self.writer.add_scalar('Train/2d_joint_loss', loss.detach(), itr + epoch * len(self.train_loader))
            if itr % cfg.print_iter == 0:
                screen = ['[Epoch %d/%d]' % (epoch+1, self.args.epoch),
                        '[Batch %d/%d]' % (itr, len(self.train_loader)),
                        '[lr %f]' % (self.get_lr())]
                screen += ['[2d_joint_loss: %.4f]' % (loss.detach())]
                self.logger.info(''.join(screen))

            timer['statistics'] += self.timer.toc()
        self.scheduler.step()
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.logger.info('Training Time consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

    def test(self, epoch):
        result = evaluate_joints(self.model, self.val_loader)
        JointErr = result['JointErr']

        self.writer.add_scalar('Val/JointErr', JointErr, epoch + 1)
        self.writer.add_scalar('Val/PCK@5', result['PCK@5'], epoch + 1)

        self.save_model({
            'epoch': epoch + 1,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr': self.get_lr()
        }, 'last')
        if JointErr < self.min_joint_err:
            self.save_model({
                'epoch': epoch + 1,
                'network': self.model.state_dict(),
            }, "best")
            self.min_joint_err = JointErr
            self.early_stop = 0
        else:
            self.early_stop += 1

        self.logger.info("[epoch: {}]".format(epoch))
        for k, v in result.items():
            self.logger.info("[%s: %.2f]" % (k, v))
        self.logger.info("[min_JointErr: %.2f]" % (self.min_joint_err))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=cfg.data_root)
    parser.add_argument('--output-root', default=cfg.output_root)
    parser.add_argument('--checkpoint', default=cfg.checkpoint, help='original ViTPose-B weights')
    parser.add_argument('--continue-train', action='store_true', default=cfg.continue_train)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--num-workers', type=int, default=cfg.num_workers)
    parser.add_argument('--seq-gap', type=int, default=cfg.seq_gap)
    parser.add_argument('--early-stop', type=int, default=cfg.early_stop)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    if args.device is not None:
        cfg.device = torch.device(args.device)
    cfg.data_root = args.data_root
    cfg.num_workers = args.num_workers
    cfg.batch_size = args.batch_size
    cfg.epoch = args.epoch
    args.experiment = cfg.experiment_name() + '-single'

    cudnn.benchmark = True

    process = processer(args)
    process.make_model()
    process.load_data()

    process.start_train()
