TRAIN_DIR = './CRNN/CAPTCHA/train'
VAL_DIR = './CRNN/CAPTCHA/val'

BATCH_SIZE = 8
N_WORKERS = 0

CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789'
VOCAB_SIZE = len(CHARS) + 1

lr = 0.02
weight_decay = 1e-5
momentum = 0.7

EPOCHS = 10