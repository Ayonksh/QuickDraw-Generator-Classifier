# common parameters
IMG_SIZE = 32
TRAIN_EPOCH = 30
NUM_CLASS = 20
NAME_CLASS = ["airplane", "alarm clock", "apple", "banana", "bed",
           "bicycle", "bird", "birthday cake", "book", "camera",
           "candle", "carrot", "chair", "cloud", "cup",
           "door", "ear", "eye", "fish", "hammer"]

# cdcgan training parameters
BATCH_SIZE = 32
SAMPLE_SIZE = 20000
CDCGAN_LR = 0.0002
DEPTH = 128

# resnet34 training parameters
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 64
RESNET_LR = 0.1
LR_DECAY_STEP = [12, 20]
GAMMA = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
