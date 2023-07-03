class Param(object):
    def __init__(self):
        # Path
        self.ROOT = 'D:/AI_CONNECT/'
        self.DATASET_PATH = f'{self.ROOT}/DB'
        self.OUTPUT_CKP = f'{self.ROOT}/backup/swin_small_patch4_window7_224/try_high_pass_filter'
        self.OUTPUT_LOSS = f'{self.ROOT}/backup/swin_small_patch4_window7_224/try_high_pass_filter/log'
        self.CKP_LOAD = False

        # Data
        self.DATA_STYPE = ['train', 'test']
        self.DATA_CLS = ['real_images', 'fake_images']
        self.SIZE = 224

        # Train or Test
        self.k = 5
        self.MAX_EPOCH = 10
        self.LR = 1e-4
        self.BATCHSZ = 32

        # Handler
        # run_type 0 : train, 1 : test
        self.run_type = 0