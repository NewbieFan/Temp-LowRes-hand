import torch

class Config:
    head = 'SAR'
    backbone = '3Dmap16-globalvec-rootrel(1)'
    dataset = 'DexYCB'
    testset = 'DexYCB'
    output_root = './output'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # network
    depth_box = 0.2
    num_vert = 778
    seqembed_times = 3
    time_steps = 5
    seq_gap = 4
    # training
    num_workers = 16
    batch_size = 64
    lr = 2e-4 
    epoch = 30
    print_iter = 100
    early_stop = 15
    input_img_shape = (64,64)
    # data augment
    downsaug = True
    downsratio = 64
    # -------------
    data_root = './data/DexYCB'
    checkpoint = ''
    continue_train = False
    s_downs = f'_low-res{downsratio}' if downsaug else ''
    test_name = f'{testset}_{input_img_shape}' + s_downs

    def experiment_name(self):
        return f'{self.backbone}-{self.seqembed_times}-{self.head}'+ f'_{self.dataset}_{self.epoch}' + f'_{self.input_img_shape}'+self.s_downs

cfg = Config()
pass
