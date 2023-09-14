from functional                    import TrainEpoch, ValidEpoch, check_predictions, draw_figure
from datagen                       import dataloader_setting
from sklearn.metrics.pairwise      import cosine_similarity
from torch.utils.tensorboard       import SummaryWriter
from statistics                    import mean, median
from config                        import *
from scipy                         import sparse
from tqdm                          import trange
import segmentation_models_pytorch as smp
import torch.optim                 as optim
import matplotlib.pyplot           as plt
import networkx                    as nx
import numpy                       as np
import torch
import cv2
import os


#############################################################################
#############################################################################

# Representativeness

#############################################################################
#############################################################################


class ExtractCosineSimilarity(object):
    def __init__(self, model, img_paths, batch_size, patch_size, threshold):
        self.encoder    = model.__dict__['_modules']['encoder']
        self.img_paths  = img_paths
        self.batch_size = batch_size 
        self.patch_size = patch_size
        self.threshold  = threshold 

    def make_image_array(self):
        '''
        make image feature array
        '''
        init_list = list()
        for batch in range(0, len(self.img_paths), self.batch_size):
            batch_path  = self.img_paths[batch:min(batch+self.batch_size,len(self.img_paths))]
            image_batch = np.zeros((len(batch_path), self.patch_size, self.patch_size, 3))
            for i, img_path in enumerate(batch_path):
                image_batch[i,...] = cv2.imread(img_path)/255
            image_tensor = self.encoder(torch.from_numpy(image_batch).float().to(DEVICE).permute(0,3,1,2))[1]
            gap_tensor   = torch.mean(image_tensor.view(image_tensor.size(0), image_tensor.size(1), -1), dim=2).squeeze().cpu().detach().numpy()
            batch_array  = [x for x in gap_tensor]
            init_list.append(batch_array)
        final_list  = sum(init_list, [])
        image_array = np.array(final_list)
        return image_array

    def binary_matrix(self):
        '''
        calculate cosine similarity matrix from image feature array
        '''
        image_array = self.make_image_array()
        sparse_matrix = sparse.csr_matrix(image_array)
        cossim_matrix = cosine_similarity(sparse_matrix)
        binary_matrix = np.where(cossim_matrix>=self.threshold,1,0)
        return binary_matrix

    def maximun_cover_index(self):
        '''
        calculate maximmum cover index
        '''
        binary_matrix = self.binary_matrix()
        graph = nx.to_networkx_graph(binary_matrix)
        node_index_list = []
        while list(graph.nodes):
            node = max(graph.nodes, key=lambda x:len(list(graph.adj[x])))
            edge = list(graph.adj[node])
            if len(edge) == 0:
                node_index_list.append(node) 
                graph.remove_node(node)
            else:
                node_index_list.append(node)      
                graph.remove_nodes_from(edge)
        print(len(node_index_list))
        return node_index_list


#############################################################################
#############################################################################

# Estimate Uncertainty

#############################################################################
#############################################################################


class BatchTTA(object):
    def __init__(self, model, device, batch_size, patch_size):
        self.model      = model
        self.device     = device
        self.batch_size = batch_size
        self.patch_size = patch_size

    def predict(self, img_patch, adjust):
        img_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3))
        for i, img in enumerate(img_patch):
            img_batch[i,...] = self.augmentation(img, adjust)
        
        x_tensor = torch.from_numpy(img_batch).float().to(self.device).permute(0,3,1,2)
        pr_mask = self.model(x_tensor).squeeze().cpu().detach().numpy()
        
        msk_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size))
        for j, msk in enumerate(pr_mask):
            msk_batch[j,...] = self.reverse(msk, adjust)
        
        return msk_batch
                             
    def augmentation(self, image, adjust):
        if adjust == 'flip_ud':
            return np.flipud(image)
        elif adjust == 'flip_lr':
            return np.fliplr(image)
        elif adjust == 'ROT90':
            return np.rot90(image, 1)
        elif adjust == 'ROT180':
            return np.rot90(image, 2)
        elif adjust == 'ROT270':
            return np.rot90(image, 3)
        else:
            return image
        
    def reverse(self, image, adjust):
        if adjust == 'flip_ud':
            return np.flipud(image)
        elif adjust == 'flip_lr':
            return np.fliplr(image)
        elif adjust == 'ROT90':
            return np.rot90(image, -1)
        elif adjust == 'ROT180':
            return np.rot90(image, -2)
        elif adjust == 'ROT270':
            return np.rot90(image, -3)
        else:
            return image        

    def apply_tta(self, img_patch):
        return np.mean([self.predict(img_patch, 'no'), self.predict(img_patch, 'flip_ud'), self.predict(img_patch, 'flip_lr'), self.predict(img_patch, 'ROT90'), self.predict(img_patch, 'ROT180'), self.predict(img_patch, 'ROT270')], axis=0)

    def no_tta(self, img_patch):
        return np.mean([self.predict(img_patch, 'no'), self.predict(img_patch, 'no'), self.predict(img_patch, 'no'), self.predict(img_patch, 'no'), self.predict(img_patch, 'no'), self.predict(img_patch, 'no')], axis=0)


#############################################################################
#############################################################################

# Select Pseudo Label

#############################################################################
#############################################################################


class SelectUnlabel():
    def __init__(self, model, unlabel_zip, iteration, save_path):
        self.model       = model
        self.unlabel_zip = unlabel_zip
        self.iteration   = iteration
        self.save_path   = save_path

    def test_time_dropout(self):
        self.model.eval()
        for each_module in self.model.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

    def path_to_image_mask(self, path_list):
        image_batch = np.zeros((len(path_list),) + IMAGE_SIZE + (3,))
        mask_batch  = np.zeros((len(path_list),) + IMAGE_SIZE)
        for i, path in enumerate(path_list):
            image_batch[i] = cv2.imread(path[0])/255
            mask_batch[i]  = cv2.imread(path[1],0)
        return image_batch, mask_batch

    def save_figure(self, number, description, uncertainty,*images):
        os.makedirs(f'{self.save_path}/{description}_{self.iteration}/', exist_ok = True)
        fig_path  = f'{self.save_path}/{description}_{self.iteration}/{number}.png'
        plt.figure(figsize=(5*len(images),5))
        for th, image in enumerate(images):
            plt.subplot(1,len(images),th+1)
            plt.imshow(image, vmin=0)
            plt.axis('off')
        plt.title(uncertainty)
        plt.savefig(fig_path, dpi=300)
        plt.cla()    

    def entropy_function(self, mask):
        entropy_func = lambda x: -1 * np.sum(np.log(x + np.finfo(np.float32).eps) * x, axis=2)
        uncer_map    = np.expand_dims(entropy_func(mask), axis=2)
        uncertainty  = round(uncer_map.sum())
        return uncer_map, uncertainty

    def compute_batch_entropy(self, iter_tta, image_batch, mask_batch, results_batch):
        entropy_batch = []
        for each, (image, mask, results) in enumerate(zip(image_batch, mask_batch, results_batch)):
            results                = np.expand_dims(results, axis=2)
            fig_num                = iter_tta+each
            uncer_map, uncertainty = self.entropy_function(results)
            entropy_batch.append(uncertainty)
        self.save_figure(fig_num, 'uncertain', uncertainty, image, mask, results.round(), uncer_map)
        return entropy_batch  

    def compute_entropy(self):
        # for test time dropout & test time augmentation
        self.test_time_dropout()
        # compute entropy
        entropy_list = list()
        for iter_tta in trange(0, len(self.unlabel_zip), BATCH_SIZE):
            path_list               = self.unlabel_zip[iter_tta:iter_tta+BATCH_SIZE]
            image_batch, mask_batch = self.path_to_image_mask(path_list)
            batch_tta               = BatchTTA(self.model, DEVICE, len(image_batch), IMAGE_SIZE[0])
            results_batch           = batch_tta.apply_tta(image_batch)
            results_batch           = batch_tta.no_tta(image_batch)
            entropy_batch           = self.compute_batch_entropy(iter_tta, image_batch, mask_batch, results_batch)
            entropy_list.extend(entropy_batch)
        return entropy_list

    def argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def get_pseudo_label(self, entropy_list):
        raise NotImplementedError

    def select_unlabel(self, entropy_list):
        raise NotImplementedError


class ALSelectUnlabel(SelectUnlabel):
    def __init__(self, model, unlabel_zip, iteration, save_path):
        super().__init__(
            model       = model,
            unlabel_zip = unlabel_zip,
            iteration   = iteration,
            save_path   = save_path
        )

    def select_unlabel(self, entropy_list):
        # sort index by uncertainty & select NB_ANNOTATION unlabeled path zip of NB_ANNOTATION size
        entropy_index = self.argsort(entropy_list)[::-1][:NB_ANNOTATION]
        final_zip     = [self.unlabel_zip[th] for th in entropy_index]
        return final_zip


class ALRSelectUnlabel(SelectUnlabel):
    def __init__(self, model, unlabel_zip, iteration, save_path, threshold_sim):
        super().__init__(
            model       = model,
            unlabel_zip = unlabel_zip,
            iteration   = iteration,
            save_path   = save_path
        )
        self.threshold_sim = threshold_sim

    def select_unlabel(self, entropy_list):
        # sort index by uncertainty & select NB_ANNOTATION unlabeled image path of NB_ANNOTATION*5 size
        entropy_index  = self.argsort(entropy_list)[::-1][:NB_ANNOTATION*5]
        entropy_zip    = [self.unlabel_zip[th] for th in entropy_index]
        entropy_image  = [self.unlabel_zip[th][0] for th in entropy_index]
        # sort index by similarity & select NB_ANNOTATION unlabeled path zip of NB_ANNOTATION size
        uncertant_ecs  = ExtractCosineSimilarity(self.model, entropy_image, BATCH_SIZE, IMAGE_SIZE[0], self.threshold_sim)
        ecs_index      = uncertant_ecs.maximun_cover_index()[:NB_ANNOTATION]
        ecs_zip        = [entropy_zip[th] for th in ecs_index]
        rest_zip       = list(set(entropy_zip)-set(ecs_zip))
        final_zip      = ecs_zip + rest_zip[:max(NB_ANNOTATION-len(ecs_zip),0)]
        return final_zip, ecs_zip


class CEALSelectUnlabel(SelectUnlabel):
    def __init__(self, model, unlabel_zip, iteration, save_path, threshold_en):
        super().__init__(
            model       = model,
            unlabel_zip = unlabel_zip,
            iteration   = iteration,
            save_path   = save_path
        )
        self.threshold_en  = threshold_en

    def get_pseudo_label(self, entropy_list):
        certain_index = [i for i, entropy in enumerate(entropy_list) if entropy < self.threshold_en]
        certain_image = [self.unlabel_zip[th][0] for th in certain_index]
        certain_mask  = [self.unlabel_zip[th][1] for th in certain_index]
        certain_entropy = [entropy_list[th] for th in certain_index]
        certain_zip   = list()
        os.makedirs(f'{self.save_path}/mask_{self.iteration}', exist_ok = True)
        for k, image_path in enumerate(certain_image):
            image     = cv2.imread(image_path)/255
            gt_mask   = cv2.imread(certain_mask[k],0)
            entropy   = certain_entropy[k]
            tensor    = torch.from_numpy(image).float().to(DEVICE).unsqueeze(0).permute(0,3,1,2)
            pr_mask   = self.model(tensor)
            pr_mask   = pr_mask.squeeze().cpu().detach().numpy().round()
            self.save_figure(k, 'certain', entropy, image, gt_mask, pr_mask)
            mask_path = f'{self.save_path}/mask_{self.iteration}/{k}.png'
            cv2.imwrite(mask_path, pr_mask)
            certain_zip.append((image_path, mask_path))
        return certain_zip

    def select_unlabel(self, entropy_list):
        # sort index by uncertainty & select NB_ANNOTATION unlabeled image path of NB_ANNOTATION*5 size
        entropy_index = self.argsort(entropy_list)[::-1][:NB_ANNOTATION]
        final_zip     = [self.unlabel_zip[th] for th in entropy_index]
        return final_zip


class CEALRSelectUnlabel(SelectUnlabel):
    def __init__(self, model, unlabel_zip, iteration, save_path, threshold_en, threshold_sim):
        super().__init__(
            model       = model,
            unlabel_zip = unlabel_zip,
            iteration   = iteration,
            save_path   = save_path
        )
        self.threshold_en = threshold_en
        self.threshold_sim = threshold_sim

    def get_pseudo_label(self, entropy_list):
        certain_index   = [i for i, entropy in enumerate(entropy_list) if entropy < self.threshold_en]
        certain_image   = [self.unlabel_zip[th][0] for th in certain_index]
        certain_mask    = [self.unlabel_zip[th][1] for th in certain_index]
        certain_entropy = [entropy_list[th] for th in certain_index]
        certain_zip     = list()
        os.makedirs(f'{self.save_path}/mask_{self.iteration}', exist_ok = True)
        for k, image_path in enumerate(certain_image):
            image     = cv2.imread(image_path)/255
            gt_mask   = cv2.imread(certain_mask[k],0)
            entropy   = certain_entropy[k]
            tensor    = torch.from_numpy(image).float().to(DEVICE).unsqueeze(0).permute(0,3,1,2)
            pr_mask   = self.model(tensor)
            pr_mask   = pr_mask.squeeze().cpu().detach().numpy().round()
            self.save_figure(k, 'certain', entropy, image, gt_mask, pr_mask)
            mask_path = f'{self.save_path}/mask_{self.iteration}/{k}.png'
            cv2.imwrite(mask_path, pr_mask)
            certain_zip.append((image_path, mask_path))
        return certain_zip

    def select_unlabel(self, entropy_list):
        # sort index by uncertainty & select NB_ANNOTATION unlabeled image path of NB_ANNOTATION*5 size
        entropy_index  = self.argsort(entropy_list)[::-1][:NB_ANNOTATION*5]
        entropy_zip    = [self.unlabel_zip[th] for th in entropy_index]
        entropy_image  = [self.unlabel_zip[th][0] for th in entropy_index]
        # sort index by similarity & select NB_ANNOTATION unlabeled path zip of NB_ANNOTATION size
        uncertant_ecs  = ExtractCosineSimilarity(self.model, entropy_image, BATCH_SIZE, IMAGE_SIZE[0], self.threshold_sim)
        ecs_index      = uncertant_ecs.maximun_cover_index()[:NB_ANNOTATION]
        ecs_zip        = [entropy_zip[th] for th in ecs_index]
        rest_zip       = list(set(entropy_zip)-set(ecs_zip))
        final_zip      = ecs_zip + rest_zip[:max(NB_ANNOTATION-len(ecs_zip),0)]
        return final_zip, ecs_zip


#############################################################################
#############################################################################

# Active Learning Strategy

#############################################################################
#############################################################################


def train_loop(model, train_zip, test_zip, writer, iteration, f):
    # dataloader setting
    train_loader, test_loader = dataloader_setting(train_zip, test_zip)
    
    # loss, metrics, optimizer and schduler setting
    loss = getattr(smp.utils.losses, LOSS)()
    metrics = [smp.utils.metrics.IoU(), smp.utils.metrics.Fscore(),]
    optimizer = getattr(optim, OPTIMIZER)(params=model.parameters(), lr=LR)

    # epoch setting
    model.train()
    train_epoch = TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=DEVICE, verbose=True)
    model.eval()
    valid_epoch = ValidEpoch(model, loss=loss, metrics=metrics, device=DEVICE, verbose=True)
    
    nb_fine_tune = NB_INITIAL_EPOCHS if iteration == 0 else NB_FINE_TUNE

    for current_epoch in range(0, nb_fine_tune):
        f.write(f'\n    Fine Tuning Epoch: {current_epoch}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(test_loader)
        val_loss = round(valid_logs['loss'],4); val_iou = round(valid_logs['iou_score'],4); val_f1 = round(valid_logs['fscore'],4)
        f.write(f'\n        Main Epoch: {iteration}, Tuning Epoch: {current_epoch}, Loss: {val_loss}, IoU: {val_iou}, Fscore:{val_f1}\n')

    if iteration != 0:
        xs, ys = next(iter(test_loader))
        figure = check_predictions(BATCH_SIZE, DEVICE, model, xs, ys)
        writer.add_scalars('Loss', {'test_loss':val_loss}, iteration)
        writer.add_scalars('IoU', {'test_iou':val_iou}, iteration)
        writer.add_scalars('Fscore', {'test_fscore':val_f1}, iteration)
        writer.add_figure('check predictions !', figure, global_step=iteration)
    return model


def init_training(model, f ):
    f.write(f'\n\nInitial Training\n\nInit Epoch: 0')
    init_zip   = TRAIN_ZIP[:NB_LABELED]
    init_model = train_loop(model, init_zip, TEST_ZIP, None, 0, f)
    torch.save(init_model, INIT_MODEL_PATH)    


def random_learning(model, f):
    # log
    writer = SummaryWriter(log_dir=f'{TESORBOARD_PATH}/{TOT_DESCRIPTION}_{DESCRIPTION_RAN}', filename_suffix=DESCRIPTION_RAN)   
    f.write('\n\nRandom Learning\n\n')
    
    for iteration in range(1, NB_ACTIVE_EPOCHS):
        # labeling
        labeled_zip = TRAIN_ZIP[:NB_LABELED+NB_ANNOTATION*iteration]
        
        # log
        f.write(f'\nRandom Learning Epoch: {iteration}\n    Labeled Zip Size: {len(labeled_zip)}\n')
        
        # random learning
        model = train_loop(model, labeled_zip, TEST_ZIP, writer, iteration, f)
        torch.save(model, f'{MODEL_SAVE_PATH}/{iteration}_{DESCRIPTION_RAN}.pth')
    torch.save(model, RR_MODEL_PATH)



def active_learning(model, f):
    labeled_zip = TRAIN_ZIP[:NB_LABELED]
    unlabel_zip = TRAIN_ZIP[NB_LABELED:]

    writer = SummaryWriter(log_dir=f'{TESORBOARD_PATH}/{TOT_DESCRIPTION}_{DESCRIPTION_AL}', filename_suffix=DESCRIPTION_AL)   
    f.write('\n\nActive Learning\n\n')
    
    for iteration in range(1, NB_ACTIVE_EPOCHS):
        # labeling
        unlabel_class   = ALSelectUnlabel(model, unlabel_zip, iteration, AL_SAVE_PATH)
        entropy_list    = unlabel_class.compute_entropy()
        new_labeled_zip = unlabel_class.select_unlabel(entropy_list)
        labeled_zip    += new_labeled_zip
        unlabel_zip     = list(set(unlabel_zip)-set(new_labeled_zip))

        # log
        f.write(f'\nActive Learning Epoch: {iteration}')
        f.write(f'    Labeled Zip Size: {len(labeled_zip)}, Unlabel Zip Size: {len(unlabel_zip)}\n')
        f.write(f'    entropy_min: {min(entropy_list)}, entropy_max: {max(entropy_list)}, entropy_mean: {round(mean(entropy_list))}, entropy_median: {round(median(entropy_list))}')

        # active learning
        model = train_loop(model, labeled_zip, TEST_ZIP, writer, iteration, f)
        torch.save(model, f'{MODEL_SAVE_PATH}/{iteration}_{DESCRIPTION_AL}.pth')
    torch.save(model, AL_MODEL_PATH)
    draw_figure('uncertain', AL_SAVE_PATH, writer)



def active_learning_representive(model, f):
    labeled_zip = TRAIN_ZIP[:NB_LABELED]
    unlabel_zip = TRAIN_ZIP[NB_LABELED:]

    writer = SummaryWriter(log_dir=f'{TESORBOARD_PATH}/{TOT_DESCRIPTION}_{DESCRIPTION_ALR}', filename_suffix=DESCRIPTION_ALR)   
    f.write('\n\nActive Learning Representive\n\n')
    
    for iteration in range(1, NB_ACTIVE_EPOCHS):
        # labeling
        threshold_sim            = THRESHOLD_SIM
        unlabel_class            = ALRSelectUnlabel(model, unlabel_zip, iteration, ALR_SAVE_PATH, threshold_sim)
        entropy_list             = unlabel_class.compute_entropy()
        new_labeled_zip, ecs_zip = unlabel_class.select_unlabel(entropy_list)
        labeled_zip             += new_labeled_zip
        unlabel_zip              = list(set(unlabel_zip)-set(new_labeled_zip))

        # log
        f.write(f'\nActive Learning Represetive Epoch: {iteration}, cosine similarity threshold: {threshold_sim}')
        f.write(f'    Labeled Zip Size: {len(labeled_zip)}, Coverset Zip Size: {len(ecs_zip)}, Unlabel Zip Size: {len(unlabel_zip)}\n')
        f.write(f'    entropy_min: {min(entropy_list)}, entropy_max: {max(entropy_list)}, entropy_mean: {round(mean(entropy_list))}, entropy_median: {round(median(entropy_list))}')
        
        # active learning
        model = train_loop(model, labeled_zip, TEST_ZIP, writer, iteration, f)
        torch.save(model, f'{MODEL_SAVE_PATH}/{iteration}_{DESCRIPTION_ALR}.pth')
    torch.save(model, ALR_MODEL_PATH)
    draw_figure('uncertain', ALR_SAVE_PATH, writer)



def ceal(model, f):
    labeled_zip = TRAIN_ZIP[:NB_LABELED]
    unlabel_zip = TRAIN_ZIP[NB_LABELED:]

    writer = SummaryWriter(log_dir=f'{TESORBOARD_PATH}/{TOT_DESCRIPTION}_{DESCRIPTION_CEAL}', filename_suffix=DESCRIPTION_CEAL)   
    f.write('\n\nCost Effective Active Learning\n\n')
    
    for iteration in range(1, NB_ACTIVE_EPOCHS):
        # labeling
        threshold_en    = THRESHOLD_EN
        unlabel_class   = CEALSelectUnlabel(model, unlabel_zip, iteration, CEAL_SAVE_PATH, threshold_en)
        entropy_list    = unlabel_class.compute_entropy()
        new_labeled_zip = unlabel_class.select_unlabel(entropy_list)
        pseduo_label_zip= unlabel_class.get_pseudo_label(entropy_list)
        labeled_aux_zip = new_labeled_zip + pseduo_label_zip + labeled_zip
        labeled_zip    += new_labeled_zip
        unlabel_zip     = list(set(unlabel_zip)-set(new_labeled_zip))

        # log
        f.write(f'\nCost Effective Active Learning Epoch: {iteration}, entropy threshold: {threshold_en}')
        f.write(f'    Labeled Zip Size: {len(labeled_zip)}, Pseudo Label Zip Size: {len(pseduo_label_zip)}, Unlabel Zip Size: {len(unlabel_zip)}\n')
        f.write(f'    entropy_min: {min(entropy_list)}, entropy_max: {max(entropy_list)}, entropy_mean: {round(mean(entropy_list))}, entropy_median: {round(median(entropy_list))}')

        # active learning
        model = train_loop(model, labeled_aux_zip, TEST_ZIP, writer, iteration, f)
        torch.save(model, f'{MODEL_SAVE_PATH}/{iteration}_{DESCRIPTION_CEAL}.pth')
    torch.save(model, CEAL_SAVE_PATH)
    draw_figure('uncertain', CEAL_SAVE_PATH, writer)
    draw_figure('certain', CEAL_SAVE_PATH, writer)



def ceal_representive(model, f):
    labeled_zip = TRAIN_ZIP[:NB_LABELED]
    unlabel_zip = TRAIN_ZIP[NB_LABELED:]

    writer = SummaryWriter(log_dir=f'{TESORBOARD_PATH}/{TOT_DESCRIPTION}_{DESCRIPTION_CEALR}', filename_suffix=DESCRIPTION_CEALR)   
    f.write('\n\nCost Effective Active Learning Representative\n\n')
    
    for iteration in range(1, NB_ACTIVE_EPOCHS):
        # labeling
        threshold_en             = THRESHOLD_EN*((1-DECREASE_RATE_EN)**(iteration-1))
        threshold_sim            = THRESHOLD_SIM
        unlabel_class            = CEALRSelectUnlabel(model, unlabel_zip, iteration, CEALR_SAVE_PATH, threshold_en, threshold_sim)
        entropy_list             = unlabel_class.compute_entropy()
        new_labeled_zip, ecs_zip = unlabel_class.select_unlabel(entropy_list)
        pseduo_label_zip         = unlabel_class.get_pseudo_label(entropy_list)
        labeled_aux_zip          = new_labeled_zip + pseduo_label_zip + labeled_zip
        labeled_zip             += new_labeled_zip
        unlabel_zip              = list(set(unlabel_zip)-set(new_labeled_zip))

        # log
        f.write(f'\nCost Effective Active Learning Representative Epoch: {iteration}, cosine similarity threshold: {threshold_sim}, entropy threshold: {threshold_en}')
        f.write(f'    Labeled Zip Size: {len(labeled_zip)}, Coverset Zip Size: {len(ecs_zip)}, Pseudo Label Zip Size: {len(pseduo_label_zip)}, Unlabel Zip Size: {len(unlabel_zip)}\n')
        f.write(f'    entropy_min: {min(entropy_list)}, entropy_max: {max(entropy_list)}, entropy_mean: {round(mean(entropy_list))}, entropy_median: {round(median(entropy_list))}')
        
        # active learning
        model = train_loop(model, labeled_aux_zip, TEST_ZIP, writer, iteration, f)
        torch.save(model, f'{MODEL_SAVE_PATH}/{iteration}_{DESCRIPTION_CEALR}.pth')
    torch.save(model, CEALR_SAVE_PATH)
    draw_figure('uncertain', CEALR_SAVE_PATH, writer)
    draw_figure('certain', CEALR_SAVE_PATH, writer)