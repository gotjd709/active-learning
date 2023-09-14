import pickle
import torch

## Train Base Setting
SEED              = 12345
MODEL             = 'DeepLabV3Plus'
ENCODER_NAME      = 'resnet18'
ENCODER_WEIGHTS   = None
ACTIVATION        = 'sigmoid'
CLASSES           = 1
DEVICE            = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE        = (512,512)
NUM_WORKER        = 0
LOSS              = 'DiceLoss'
OPTIMIZER         = 'Adam'
LR                = 1e-5
SHUFFLE           = True

## Active Learning Setting
NB_FINE_TUNE      = 2
NB_ACTIVE_EPOCHS  = 22
DECREASE_RATE_EN  = 0.025
NB_INITIAL_EPOCHS = 9
THRESHOLD_EN      = 5000
THRESHOLD_SIM     = 0.9999
DECREASE_RATE_SIM = 0.0001
NB_LABELED        = 226
NB_ANNOTATION     = 99
BATCH_SIZE        = 128

## Descript Setting
TOT_DESCRIPTION   = f'descript_setting'

## Log Setting
TESORBOARD_PATH   = f'log_path/tensorboard/'
LOG_SAVE_PATH     = f'logt_path/{TOT_DESCRIPTION}'

## Stragety Setting 
DESCRIPTION_RAN   = 'random'
DESCRIPTION_AL    = 'active_learning'
DESCRIPTION_ALR   = 'active_learning_representive'
DESCRIPTION_CEAL  = 'ceal'
DESCRIPTION_CEALR = 'ceal_representive'

## Model Path Setting
MODEL_SAVE_PATH   = f'model_path/{TOT_DESCRIPTION}'
INIT_MODEL_PATH   = f'{MODEL_SAVE_PATH}_init.pth'
RR_MODEL_PATH     = f'{MODEL_SAVE_PATH}_{DESCRIPTION_RAN}.pth'
AL_MODEL_PATH     = f'{MODEL_SAVE_PATH}_{DESCRIPTION_AL}.pth'
ALR_MODEL_PATH    = f'{MODEL_SAVE_PATH}_{DESCRIPTION_ALR}.pth'
CEAL_MODEL_PATH   = f'{MODEL_SAVE_PATH}_{DESCRIPTION_CEAL}.pth'
CEALR_MODEL_PATH  = f'{MODEL_SAVE_PATH}_{DESCRIPTION_CEALR}.pth'

## Data Path Setting
DATA_SAVE_PATH    = f'data_path/{TOT_DESCRIPTION}'
AL_SAVE_PATH      = f'{DATA_SAVE_PATH}_{DESCRIPTION_AL}'
ALR_SAVE_PATH     = f'{DATA_SAVE_PATH}_{DESCRIPTION_ALR}'
CEAL_SAVE_PATH    = f'{DATA_SAVE_PATH}_{DESCRIPTION_CEAL}'
CEALR_SAVE_PATH   = f'{DATA_SAVE_PATH}_{DESCRIPTION_CEALR}'

## Data Load
with open('./cm16_5x_train.pickle', 'rb') as fr:
    TRAIN_ZIP = pickle.load(fr)

with open('./cm16_5x_test.pickle', 'rb') as fr:
    TEST_ZIP = pickle.load(fr)