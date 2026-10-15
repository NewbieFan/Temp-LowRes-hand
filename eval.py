import os
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from utils.logging import get_logger

from models.vithand import getSarMultiSeqModel
from datasets.dexycb import DexYCB_Temporal
from engine import evaluate

from config import cfg


class processer(object):
    def __init__(self, args):
        self.args = args
        self.device = cfg.device
        self.output_dir = os.path.join(self.args.output_root, self.args.experiment, 'test')
        log_folder = os.path.join(self.output_dir, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'test_' + self.args.test_name + '.log')
        self.logger = get_logger(log_file=logfile, name="Test")
        self.logger.info(f'Start testing: %s' % ('Test_' + self.args.test_name))

    def make_model(self):
        self.model = getSarMultiSeqModel().to(self.device)

        ckpt_path = self.args.checkpoint or \
            os.path.join(self.args.output_root, self.args.experiment, 'train/checkpoints/ckpt_best.pth')
        ckpt = torch.load(ckpt_path, map_location='cpu')

        missing, unexpected = self.model.load_state_dict(ckpt['network'], strict=False)
        self.logger.info(f'Load Model weight from checkpoint:{ckpt_path}')
        if missing or unexpected:
            self.logger.info('[missing_keys: {}]'.format(len(missing)))
            self.logger.info('[unexpected_keys: {}]'.format(len(unexpected)))
            for k in missing[:5]:
                self.logger.info('   missing %s' % k)
            for k in unexpected[:5]:
                self.logger.info('   unexpected %s' % k)
        self.logger.info("The model ViTHand is made successfully.")

    def load_data(self):
        self.test_set = DexYCB_Temporal(self.args.data_root, self.args.split,
                                        T=self.args.time_steps, gap=self.args.seq_gap)
        self.test_loader = DataLoader(self.test_set,
                                      batch_size=self.args.batch_size,
                                      shuffle=False,
                                      num_workers=self.args.num_workers,
                                      pin_memory=True,
                                      drop_last=False)

    def test(self):
        result = evaluate(self.model, self.test_loader, self.test_set)
        for k, v in result.items():
            self.logger.info(f"[{k}: {v:.2f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=cfg.data_root)
    parser.add_argument('--output-root', default=cfg.output_root)
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--split', default='test')
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size)
    parser.add_argument('--num-workers', type=int, default=cfg.num_workers)
    parser.add_argument('--time-steps', type=int, default=cfg.time_steps)
    parser.add_argument('--seq-gap', type=int, default=cfg.seq_gap)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    if args.device is not None:
        cfg.device = torch.device(args.device)
    cfg.data_root = args.data_root
    cfg.num_workers = args.num_workers
    cfg.batch_size = args.batch_size
    args.experiment = cfg.experiment_name()
    args.test_name = cfg.test_name

    cudnn.benchmark = True

    process = processer(args)
    process.make_model()
    process.load_data()

    process.test()
