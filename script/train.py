from util                          import init_training, random_learning, active_learning, active_learning_representive, ceal, ceal_representive
from model                         import model_setting
from config                        import *
import os

def main():
    ## model setting    
    zero_model = model_setting()

    ## log setting
    os.makedirs(MODEL_SAVE_PATH, exist_ok = True)
    f = open(LOG_SAVE_PATH, 'a')
    f.write(f'Parameter Setting \nWhether Shuffle: {SHUFFLE}, Batch Size: {BATCH_SIZE}')

    ## initial epoch
    init_training(zero_model, f)
    
    ## random learning
    init_model = torch.load(INIT_MODEL_PATH)
    random_learning(init_model, f)
    
    ## active learning
    init_model = torch.load(INIT_MODEL_PATH)
    active_learning(init_model, f)

    ## active learning Representive
    init_model = torch.load(INIT_MODEL_PATH)
    active_learning_representive(init_model, f)

    ## Cost Effective active learning
    init_model = torch.load(INIT_MODEL_PATH)
    ceal(init_model, f)

    ## Cost Effective active learning
    init_model = torch.load(INIT_MODEL_PATH)
    ceal_representive(init_model, f)

    f.close()


if __name__ == '__main__':
    main()