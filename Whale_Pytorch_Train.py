#coding=UTF-8

###Imports
import math, os, sys, time, copy, gc, cv2
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from timm.data.transforms_factory import create_transform
from timm.optim import create_optimizer_v2
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

from apex import amp

os.environ['TORCH_HOME'] = "./model/"

###Paths & Settings
INPUT_DIR = Path("./") / "data3"
OUTPUT_DIR = Path("./") / "output"

DATA_ROOT_DIR = INPUT_DIR / "convert-backfintfrecords" / "happy-whale-and-dolphin-backfin"
TRAIN_DIR = DATA_ROOT_DIR / "train_images"
TEST_DIR = DATA_ROOT_DIR / "test_images"
TRAIN_CSV_PATH = DATA_ROOT_DIR / "train.csv"
SAMPLE_SUBMISSION_CSV_PATH = DATA_ROOT_DIR / "sample_submission.csv"
PUBLIC_SUBMISSION_CSV_PATH = INPUT_DIR / "0-720-eff-b5-640-rotate" / "submission_infer_template.csv"
IDS_WITHOUT_BACKFIN_PATH = INPUT_DIR / "ids-without-backfin" / "ids_without_backfin.npy"

N_SPLITS = 5

ENCODER_CLASSES_PATH = OUTPUT_DIR / "encoder_classes.npy"
TEST_CSV_PATH = OUTPUT_DIR / "test.csv"
TRAIN_CSV_ENCODED_FOLDED_PATH = OUTPUT_DIR / "train_encoded_folded.csv"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
SUBMISSION_CSV_PATH = OUTPUT_DIR / "submission.csv"

DEBUG = False

###Configuration
r"""Models:
    convnext_xlarge_384_in22ft1k    "convNxXL22~1k384"      ☠️ max_batch_size    == [ 10(384) 55Mins/Ep ] [ 6(512) 1H35Mins/Ep ]
                                                                          AMP    == [ O1: 10(384) 30Mins/Ep ] [ O2: 20(384) 23Mins/Ep ]
    convnext_large_in22ft1k         "convNxL22~1k244"       🙀
    convnext_large_384_in22ft1k     "convNxL22~1k384"       🎃 max_batch_size == [ 16(384) 50Mins/Ep ]
    convnext_large                  "convNxL1k224"          🥶 max_batch_size == [ 16(384) 50Mins/Ep ]
    tf_efficientnetv2_xl_in21k      "effv2XL"               😡 max_batch_size == [ 10(384) 33Mins/Ep ]
    tf_efficientnet_b7_ns           "effb7NS"               🥶 max_batch_size == [ 12(384) 20Mins/Ep ]
                                                                          AMP == [ 24(384) 15Mins/Ep ]
    tf_efficientnet_b8              "effb8"                 🤢 max_batch_size == [ 12(384)??? ]
"""
CONFIG = {
    "seed": 2022,
    "epochs": 10,
    "train_img_size": 384,
    "model_name": "convnext_xlarge_384_in22ft1k",
    "model_suffix": "convNxXL22~1k384", 
    "num_classes": 15587,
    "train_batch_size": 10,
    "valid_batch_size": 10,
    "learning_rate": 3e-4,
    "scheduler": 'OneCycleLR',
    "min_lr": 1e-7,
    "T_max": 500,
    "weight_decay": 1e-6,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "test_mode":True, # enable for testing pipeline, changes epochs to 2 and uses just 100 training samples
    "enable_amp_half_precision": True, # Try it in your local machine (the code is made for working with !pip install apex, not the pytorch native apex)
    "GeM": False,
    # ArcFace Hyperparameters
    "s": 30.0, 
    "m": 0.30,
    "ls_eps": 0.0,
    "easy_margin": False
}

import logging
logging.basicConfig(level=logging.INFO,
                    filename='running.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    #format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s - %(levelname)s -:: %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"logger started. 🐳_Pytorch_model, KFold={CONFIG['n_fold']} 🔴🟡🟢 {sys.argv}")

###Prepare DataFrames
def get_image_path(id: str, dir: Path) -> str:
    return f"{dir / id}"

###SetSeed
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

###DataSets
class HappyWhaleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size, train_flag):
        self.df = df
        self.img_size = img_size

        self.image_names = self.df["image"].values
        self.image_paths = self.df["image_path"].values
        self.targets = self.df["individual_id"].values

        """
        self.transform = create_transform(
            input_size=(self.img_size, self.img_size),
            crop_pct=1.0,
        )
        """
        if train_flag:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                #A.Transpose(p=0.5),
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                A.RandomBrightnessContrast(p=0.5),  
                #A.OneOf([
                #    A.ISONoise(),   
                #    A.GaussNoise(),
                #    A.MultiplicativeNoise(elementwise=True),
                #],  p=1),  
                A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0),
                ToTensorV2()], 
                p=1.0
            )
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0),
                ToTensorV2()], 
                p=1.0
            )           
        
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_name = self.image_names[index]
        image_path = self.image_paths[index]
        img = Image.open(image_path)

        """
        if self.transform:
            img = self.transform(img)
        """
        img = np.array(img)

        if self.transform:
            img = self.transform(image=img)['image']
        

        target = self.targets[index]
        target = torch.tensor(target, dtype=torch.long)

        return {"image_name": image_name, "image": img, "target": target}

    def __len__(self) -> int:
        return len(self.df)

###ArcMargin
# From https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
# Added type annotations, device, and 16bit support
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float,
        m: float,
        easy_margin: bool,
        ls_eps: float,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

###GeM Pooling
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

###HappyWhaleModel
class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, pretrained=True, embedding_size=768):
        super(HappyWhaleModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, drop_rate=0.0)
        backbone_features = self.model.get_classifier().in_features
        self.embedding_size = embedding_size 
        if CONFIG['GeM']:
            self.model.reset_classifier(0, global_pool='')
        else:
            self.model.reset_classifier(0, global_pool='avg')
        """
        if "convnext" not in CONFIG["model_name"]:  ### None_ConvNext_Model
            self.model.classifier = nn.Identity()
            self.model.global_pool = nn.Identity()
        else:                                       ### ConvNext_Model
            self.model.reset_classifier(0, global_pool="avg")
        """
        self.gem_pool = GeM()
        """
        noNeurons = 1024
        self.linearLayer = nn.Sequential(
                    nn.Linear(backbone_features, noNeurons),
                    nn.BatchNorm1d(noNeurons),
                    nn.ReLU(),
                    nn.Dropout(p=0.2, inplace=False),

                    nn.Linear(noNeurons, self.embedding_size),
                    nn.BatchNorm1d(self.embedding_size),
                    nn.ReLU(),
                    nn.Dropout(p=0.2, inplace=False)
        ) 
        """
        self.linearLayer = nn.Sequential(
            nn.Linear(backbone_features, self.embedding_size),
        )

        self.arc = ArcMarginProduct(
                in_features=self.embedding_size, 
                out_features=CONFIG["num_classes"],
                s=CONFIG["s"], 
                m=CONFIG["m"], 
                easy_margin=CONFIG["ls_eps"], 
                ls_eps=CONFIG["ls_eps"]
 		)

        self.train_df = pd.read_csv(TRAIN_CSV_ENCODED_FOLDED_PATH)
        self.test_df = pd.read_csv(TEST_CSV_PATH)

    def forward(self, images, labels):
        features = self.model(images)
        if CONFIG['GeM']:
            pooled_features = self.gem_pool(features).flatten(1)
            emb = self.linearLayer(pooled_features)
        else:
            emb = self.linearLayer(features)
        output = self.arc(emb,labels)
        return output, emb

def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

###Freeze BatchNorm Layers
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

import Loss_Class as LC
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    #model.apply(set_bn_eval) # this will freeze the bn in training process

    dataset_size = 0
    running_loss = 0.0
    
    #train_loss_fn = LC.Cyclical_FocalLoss(epochs=CONFIG['epochs'])
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['target'].to(device, dtype=torch.long)
        
        batch_size = images.size(0)
        
        outputs, emb = model(images, labels)
        loss = criterion(outputs, labels)
        #loss = train_loss_fn(outputs, labels, epoch)
        loss = loss / CONFIG['n_accumulate']
        if (CONFIG['enable_amp_half_precision']==True):
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        del images, labels, outputs, emb
        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=f"{epoch}/{CONFIG['epochs']}", Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"Epoch-LR::\t\t\t{optimizer.param_groups[0]['lr']:e} 〽️")
    return epoch_loss

#@torch.inference_mode()
@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        labels = data['target'].to(device, dtype=torch.long)
        
        batch_size = images.size(0)

        outputs, emb = model(images, labels)
        loss = criterion(outputs, labels)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        del images, labels, outputs, emb
        bar.set_postfix(Epoch=f"{epoch}/{CONFIG['epochs']}", Valid_Loss=epoch_loss)   
    
    gc.collect()
    torch.cuda.empty_cache()
    return epoch_loss

def fetch_scheduler(optimizer, num_train_steps):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
                
    elif CONFIG['scheduler'] == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CONFIG['learning_rate'], steps_per_epoch=num_train_steps, epochs=CONFIG['epochs'])
        
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

def prepare_loaders(df, fold, img_size, stage='NOT_TRAIN'):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    if stage == 'train':
        train_dataset = HappyWhaleDataset(df_train, img_size, train_flag=True)
    else:
        train_dataset = HappyWhaleDataset(df_train, img_size, train_flag=False)
    valid_dataset = HappyWhaleDataset(df_valid, img_size, train_flag=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=22, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=22, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

def run_training(model, optimizer, scheduler, train_loader, valid_loader, device, fold):
    # To automatically log gradients
    #wandb.watch(model, log_freq=100)
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    history = defaultdict(list)

    print(f"*************************\nRuning fold==>{fold+1}/{CONFIG['n_fold']}\n*************************")
    logger.info(f"*************************\nRuning fold==>{fold+1}/{CONFIG['n_fold']}🌱🌱🌱🦋🍄🍄🍄")
    num_epochs = CONFIG['epochs']
    model_suffix = CONFIG['model_suffix']
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        logger.info(f"Epoch:{epoch}/{num_epochs}")
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch)
        
        val_epoch_loss = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        logger.info(f"Train_Epoch-{epoch}_Loss::{train_epoch_loss}, Valid_Epoch-{epoch}_Loss::{val_epoch_loss}")
        
        # Log the metrics
        #wandb.log({"Train Loss": train_epoch_loss}) 
        #wandb.log({"Valid Loss": val_epoch_loss})
        #wandb.log({"LR": optimizer.param_groups[0]['lr']})
        
        # deep copy the model
        if val_epoch_loss <= best_loss:
            print(f"Val Loss ({best_loss} ---> {val_epoch_loss} == {best_loss-val_epoch_loss:f}) 💫✨⚡️ ---> 💾:HW-{model_suffix}-fold-{fold}.bin")
            logger.info(f"Val Loss ({best_loss} ---> {val_epoch_loss} == {best_loss-val_epoch_loss:f}) 💫✨⚡️ ---> 💾:HW-{model_suffix}-fold-{fold}.bin")
            best_loss = val_epoch_loss
            #run.summary["Best Loss"] = best_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "HW-{}-fold-{}.bin".format(model_suffix, fold)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
        else:
            logger.info(f"Validation Loss no change, current epoch loss({val_epoch_loss})🪫🙀💊")
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_loss))

    logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return history


###Call Training
""" mode type: 
        0 ==> just run one specfic fold
        1 ==> run all folds
"""
def Training_and_Validation(mode=1, val_fold=0): 
    if mode == 0:
        print(f"Running_mode=🌘SINGLE_FOLD👉 \"{val_fold}\", image_size={CONFIG['train_img_size']}, model=\"{CONFIG['model_name']}\"")
        logger.info(f"Running_mode=🌘SINGLE_FOLD👉 \"{val_fold}\", image_size={CONFIG['train_img_size']}, model=\"{CONFIG['model_name']}\"")
    else:
        print(f"Running_mode=🌕FULL_FOLDS, image_size=={CONFIG['train_img_size']}, model=\"{CONFIG['model_name']}\"")
        logger.info(f"Running_mode=🌕FULL_FOLDS, image_size=={CONFIG['train_img_size']}, model=\"{CONFIG['model_name']}\"")

    for fold in range(CONFIG['n_fold']):
        if mode == 0:
            if fold != val_fold:
                continue

        model = HappyWhaleModel(CONFIG['model_name'])
        model.to(CONFIG['device']); 
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
            weight_decay=CONFIG['weight_decay'])
        if (CONFIG['enable_amp_half_precision']==True):
            opt_level = 'O1'
            model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

        train_loader, valid_loader = prepare_loaders(train_df, fold, CONFIG['train_img_size'], stage='train')
        num_train_steps = len(train_loader)
        scheduler = fetch_scheduler(optimizer, num_train_steps)

        history = run_training(model, optimizer, scheduler,
            train_loader, valid_loader,
            device=CONFIG['device'],
            fold=fold)

        del model
        gc.collect()
        torch.cuda.empty_cache()

def get_embeddings(model, dataloader, encoder, stage):
    all_image_names = []
    all_embeddings = []
    all_targets = []

    for batch in tqdm(dataloader, desc=f"Creating {stage} embeddings"):
        image_names = batch["image_name"]
        images = batch["image"].to(CONFIG['device'])
        targets = batch["target"].to(CONFIG['device'])

        outputs, embeddings = model(images, targets)

        all_image_names.append(image_names)
        all_embeddings.append(embeddings.cpu().detach().numpy())
        all_targets.append(targets.cpu().numpy())
        
        if DEBUG:
            break

    all_image_names = np.concatenate(all_image_names)
    all_embeddings = np.vstack(all_embeddings)
    all_targets = np.concatenate(all_targets)

    all_embeddings = normalize(all_embeddings, axis=1, norm="l2")
    all_targets = encoder.inverse_transform(all_targets)

    return all_image_names, all_embeddings, all_targets

###Call Inference 
import Whale_Pytorch_Inference as wpi   
@torch.no_grad()
def Inference(
    val_fold: float = 0.0,
    image_size: int = 256,
    k: int = 50,
):
    start = time.time()
    print(f"🐳🐳🐳 Inference_Model running:: fold=={val_fold}, image_size={image_size}, model=={CONFIG['model_name']}")
    model = HappyWhaleModel(CONFIG['model_name'])
    model_suffix = CONFIG['model_suffix']
    model.to(CONFIG['device']); 
    print(f"Loading pretrained model file:: ./HW-{model_suffix}-fold-{val_fold}.bin")
    logger.info(f"🐬🐬🐬 Inference_Model running:: fold=={val_fold}, image_size={image_size}, model==\"{CONFIG['model_name']}\", BIN==👉HW-{model_suffix}-fold-{val_fold}.bin👈")
    model.load_state_dict(torch.load(f'./HW-{model_suffix}-fold-{val_fold}.bin', map_location=CONFIG['device']))
    model.eval()

    CONFIG.update({"train_batch_size": 128, "valid_batch_size": 128})

    train_dl, val_dl = prepare_loaders(model.train_df, val_fold, CONFIG['train_img_size'], stage='infer')
    test_dataset = HappyWhaleDataset(model.test_df, image_size, train_flag=False)
    test_dl = DataLoader(test_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=18, shuffle=True, pin_memory=True, drop_last=False)

    encoder = wpi.load_encoder()

    train_image_names, train_embeddings, train_targets = get_embeddings(model, train_dl, encoder, stage="train")
    val_image_names, val_embeddings, val_targets = get_embeddings(model, val_dl, encoder, stage="val")
    test_image_names, test_embeddings, test_targets = get_embeddings(model, test_dl, encoder, stage="test")

    D, I = wpi.create_and_search_index(model.embedding_size, train_embeddings, val_embeddings, k)  # noqa: E741
    print("Created index with train_embeddings")

    val_targets_df = wpi.create_val_targets_df(train_targets, val_image_names, val_targets)
    print(f"val_targets_df=\n{val_targets_df.head()}")

    val_df = wpi.create_distances_df(val_image_names, train_targets, D, I, "val")
    print(f"val_df=\n{val_df.head()}")
    logger.info(f"val_df=\n{val_df.head()}")

    best_th, best_cv = wpi.get_best_threshold(val_targets_df, val_df)
    print(f"val_targets_df=\n{val_targets_df.describe()}")
    logger.info(f"val_targets_df=\n{val_targets_df.describe()}")

    train_embeddings = np.concatenate([train_embeddings, val_embeddings])
    train_targets = np.concatenate([train_targets, val_targets])
    print("Updated train_embeddings and train_targets with val data")

    D, I = wpi.create_and_search_index(model.embedding_size, train_embeddings, test_embeddings, k)  # noqa: E741
    print("Created index with train_embeddings")

    test_df = wpi.create_distances_df(test_image_names, train_targets, D, I, "test")
    print(f"test_df=\n{test_df.head()}")
    logger.info(f"test_df=\n{test_df.head()}")

    predictions = wpi.create_predictions_df(test_df, best_th)
    print(f"predictions.head()={predictions.head()}")
    
    # Fix missing predictions
    # From https://www.kaggle.com/code/jpbremer/backfins-arcface-tpu-effnet/notebook
    public_predictions = pd.read_csv(PUBLIC_SUBMISSION_CSV_PATH)
    ids_without_backfin = np.load(IDS_WITHOUT_BACKFIN_PATH, allow_pickle=True)

    ids2 = public_predictions["image"][~public_predictions["image"].isin(predictions["image"])]

    predictions = pd.concat(
        [
            predictions[~(predictions["image"].isin(ids_without_backfin))],
            public_predictions[public_predictions["image"].isin(ids_without_backfin)],
            public_predictions[public_predictions["image"].isin(ids2)],
        ]
    )
    predictions = predictions.drop_duplicates()

    predictions.to_csv(SUBMISSION_CSV_PATH, index=False)

    end = time.time()
    time_elapsed = end - start
    print('Inference complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print ("submmisson.csv generated! 💯🦀🦀")

    logger.info('Inference complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    logger.info("submmisson.csv generated! 💯🦀🦀")

###Ensembling Entry
@torch.no_grad()
def EnsembleingGroups(val_fold: float=0.0, image_size: int=256, k: int=50): 
    start = time.time()
    print(f"🐳🐳🐳 Ensemble_Model running:: fold=={val_fold}, image_size={image_size}")

    encoder = wpi.load_encoder()
    pretrained_list = [
        "HW-convNxXL22~1k384-fold-0.bin.v11",
        "HW-convNxXL22~1k384-fold-1.bin.v11",
        "HW-convNxXL22~1k384-fold-2.bin.v11",
        "HW-convNxXL22~1k384-fold-3.bin.v11",
        "HW-convNxXL22~1k384-fold-4.bin.v11",
    ]

    model_name_list = [
        "convnext_xlarge_384_in22ft1k",
        "convnext_xlarge_384_in22ft1k",
        "convnext_xlarge_384_in22ft1k",
        "convnext_xlarge_384_in22ft1k",
        "convnext_xlarge_384_in22ft1k",
        #"tf_efficientnetv2_xl_in21k",
        #"convnext_large",
    ]

    CONFIG.update({"train_batch_size": 128, "valid_batch_size": 128})

    total_test_df = None
    best_threhods = None
    for idx in range(len(pretrained_list)):
        if val_fold < 0:
            using_val_fold = idx
        else:
            using_val_fold = val_fold
        model = HappyWhaleModel(model_name_list[idx])
        model.to(CONFIG['device']); 
        print(f"Loading model {idx+1}/{len(pretrained_list)}== \"{model_name_list[idx]}\", file=={pretrained_list[idx]}, fold={using_val_fold}")
        model.load_state_dict(torch.load(f'./backup/{pretrained_list[idx]}', map_location=CONFIG['device']))
        model.eval()
        logger.info(f"🐬🐬🐬 Loading model {idx+1}/{len(pretrained_list)}::== \"{model_name_list[idx]}\", BIN:👉{pretrained_list[idx]}👈, fold={using_val_fold}")

        train_dl, val_dl = prepare_loaders(model.train_df, using_val_fold, CONFIG['train_img_size'], stage='ensemble')
        test_dataset = HappyWhaleDataset(model.test_df, image_size, train_flag=False)
        test_dl = DataLoader(test_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=18, shuffle=True, pin_memory=True, drop_last=False)
    
        train_image_names, train_embeddings, train_targets = get_embeddings(model, train_dl, encoder, stage="train")
        val_image_names, val_embeddings, val_targets = get_embeddings(model, val_dl, encoder, stage="val")
        test_image_names, test_embeddings, test_targets = get_embeddings(model, test_dl, encoder, stage="test")


        D, I = wpi.create_and_search_index(model.embedding_size, train_embeddings, val_embeddings, k)  # noqa: E741
        print("Created index with train_embeddings")

        val_targets_df = wpi.create_val_targets_df(train_targets, val_image_names, val_targets)
        print(f"val_targets_df=\n{val_targets_df.head()}")

        val_df = wpi.create_distances_df(val_image_names, train_targets, D, I, "val")
        print(f"val_df=\n{val_df.head()}")
        logger.info(f"val_df=\n{val_df.head()}")

        best_th, best_cv = wpi.get_best_threshold(val_targets_df, val_df)
        print(f"val_targets_df=\n{val_targets_df.describe()}")
        logger.info(f"val_targets_df=\n{val_targets_df.describe()}")

        train_embeddings = np.concatenate([train_embeddings, val_embeddings])
        train_targets = np.concatenate([train_targets, val_targets])
        print("Updated train_embeddings and train_targets with val data")

        D, I = wpi.create_and_search_index(model.embedding_size, train_embeddings, test_embeddings, k)  # noqa: E741
        print("Created index with train_embeddings")

        test_df = wpi.create_distances_df(test_image_names, train_targets, D, I, "test")
        print(f"test_df=\n{test_df.head()}")
        logger.info(f"test_df=\n{test_df.head()}")
        #test_df.to_csv(f"testDF-{idx}.csv")

        if total_test_df is None:
            total_test_df = test_df
        else:
            total_test_df = pd.concat([total_test_df, test_df]).reset_index(drop=True)

        if best_threhods is None:
            best_threhods = best_th
        else:
            best_threhods += best_th
    
    total_test_df_max = total_test_df.groupby(["image", "target"]).distances.max().reset_index()
    total_test_df_max = total_test_df.sort_values("distances", ascending=False).reset_index(drop=True)

    total_test_df_mean = total_test_df.groupby(["image", "target"]).distances.mean().reset_index()
    total_test_df_mean = total_test_df.sort_values("distances", ascending=False).reset_index(drop=True)

    best_threhods /= len(pretrained_list)

    predictions_max = wpi.create_predictions_df(total_test_df_max, best_threhods)
    print(f"predictions.head()={predictions_max.head()}")
    
    predictions_mean = wpi.create_predictions_df(total_test_df_mean, best_threhods)
    print(f"predictions.head()={predictions_mean.head()}")

    # Fix missing predictions
    # From https://www.kaggle.com/code/jpbremer/backfins-arcface-tpu-effnet/notebook
    public_predictions = pd.read_csv(PUBLIC_SUBMISSION_CSV_PATH)
    ids_without_backfin = np.load(IDS_WITHOUT_BACKFIN_PATH, allow_pickle=True)

    ###Inference by max value
    ids2_max = public_predictions["image"][~public_predictions["image"].isin(predictions_max["image"])]
    predictions_max = pd.concat(
        [
            predictions_max[~(predictions_max["image"].isin(ids_without_backfin))],
            public_predictions[public_predictions["image"].isin(ids_without_backfin)],
            public_predictions[public_predictions["image"].isin(ids2_max)],
        ]
    )
    predictions_max = predictions_max.drop_duplicates()
    predictions_max.to_csv(f"{SUBMISSION_CSV_PATH}-max", index=False)

    ###Inference by mean value
    ids2_mean = public_predictions["image"][~public_predictions["image"].isin(predictions_mean["image"])]
    predictions_mean = pd.concat(
        [
            predictions_mean[~(predictions_mean["image"].isin(ids_without_backfin))],
            public_predictions[public_predictions["image"].isin(ids_without_backfin)],
            public_predictions[public_predictions["image"].isin(ids2_mean)],
        ]
    )
    predictions_mean = predictions_mean.drop_duplicates()
    predictions_mean.to_csv(f"{SUBMISSION_CSV_PATH}-mean", index=False)

    end = time.time()
    time_elapsed = end - start
    print('Ensembling complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print ("submmisson.csv generated! 💯🦀🦀")

    logger.info('Ensembling complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    logger.info("submmisson.csv generated! 💯🦀🦀")

@torch.no_grad()
def EnsembleingGroups2(val_fold: float=0.0, image_size: int=256, k: int=50): 
    start = time.time()
    print(f"🐳🐳🐳 Ensemble_Model running:: fold=={val_fold}, image_size={image_size}")

    encoder = wpi.load_encoder()
    pretrained_list = [
        "HW-fold-0.bin.v6",
        "HW-effv2XL-fold-0.bin.v9",
        "HW-convNxXL22~1k384-fold-0.bin.v19",
    ]

    model_name_list = [
        "convnext_large",
        "tf_efficientnetv2_xl_in21k",
        "convnext_xlarge_384_in22ft1k",
    ]

    CONFIG.update({"train_batch_size": 128, "valid_batch_size": 128})

    tt_train_embedding = None
    tt_val_embedding = None
    tt_test_embedding = None
    for idx in range(len(pretrained_list)):
        if val_fold < 0:
            using_val_fold = idx
        else:
            using_val_fold = val_fold
        model = HappyWhaleModel(model_name_list[idx])
        model.to(CONFIG['device']); 
        print(f"Loading model {idx+1}/{len(pretrained_list)}== \"{model_name_list[idx]}\", file=={pretrained_list[idx]}, fold={using_val_fold}")
        model.load_state_dict(torch.load(f'./backup/{pretrained_list[idx]}', map_location=CONFIG['device']))
        model.eval()
        logger.info(f"🐬🐬🐬 Loading model {idx+1}/{len(pretrained_list)}::== \"{model_name_list[idx]}\", BIN:👉{pretrained_list[idx]}👈, fold={using_val_fold}")

        train_dl, val_dl = prepare_loaders(model.train_df, using_val_fold, CONFIG['train_img_size'], stage='ensemble')
        test_dataset = HappyWhaleDataset(model.test_df, image_size, train_flag=False)
        test_dl = DataLoader(test_dataset, batch_size=CONFIG['train_batch_size'], 
                              num_workers=18, shuffle=True, pin_memory=True, drop_last=False)
    
        train_image_names, train_embeddings, train_targets = get_embeddings(model, train_dl, encoder, stage="train")
        val_image_names, val_embeddings, val_targets = get_embeddings(model, val_dl, encoder, stage="val")
        test_image_names, test_embeddings, test_targets = get_embeddings(model, test_dl, encoder, stage="test")

        #tensor_train_embedding = F.normalize(torch.tensor(train_embeddings), dim=1)
        #tensor_val_embedding = F.normalize(torch.tensor(val_embeddings), dim=1)
        #tensor_test_embedding = F.normalize(torch.tensor(test_embeddings), dim=1)

        if tt_train_embedding is None:
            tt_train_embedding = train_embeddings
            tt_val_embedding = val_embeddings
            tt_test_embedding = test_embeddings
        else:
            tt_train_embedding = np.concatenate([tt_train_embedding, train_embeddings], axis=1)
            tt_val_embedding = np.concatenate([tt_val_embedding, val_embeddings], axis=1)
            tt_test_embedding = np.concatenate([tt_test_embedding, test_embeddings], axis=1)


    #train_embeddings = normalize(tt_train_embedding, axis=1)
    #val_embeddings = normalize(tt_val_embedding, axis=1)
    #test_embeddings = normalize(tt_test_embedding, axis=1)

    train_embeddings = tt_train_embedding
    val_embeddings = tt_val_embedding
    test_embeddings = tt_test_embedding
    model.embedding_size *= len(pretrained_list)

    D, I = wpi.create_and_search_index(model.embedding_size, train_embeddings, val_embeddings, k*len(pretrained_list))  # noqa: E741
    print("Created index with train_embeddings")

    val_targets_df = wpi.create_val_targets_df(train_targets, val_image_names, val_targets)
    print(f"val_targets_df=\n{val_targets_df.head()}")

    val_df = wpi.create_distances_df(val_image_names, train_targets, D, I, "val")
    print(f"val_df=\n{val_df.head()}")
    logger.info(f"val_df=\n{val_df.head()}")

    best_th, best_cv = wpi.get_best_threshold(val_targets_df, val_df)
    print(f"val_targets_df=\n{val_targets_df.describe()}")
    logger.info(f"val_targets_df=\n{val_targets_df.describe()}")

    train_embeddings = np.concatenate([train_embeddings, val_embeddings])
    train_targets = np.concatenate([train_targets, val_targets])
    print("Updated train_embeddings and train_targets with val data")

    D, I = wpi.create_and_search_index(model.embedding_size, train_embeddings, test_embeddings, k*len(pretrained_list))  # noqa: E741
    print("Created index with train_embeddings")

    test_df = wpi.create_distances_df(test_image_names, train_targets, D, I, "test")
    print(f"test_df=\n{test_df.head()}")
    logger.info(f"test_df=\n{test_df.head()}")

    predictions = wpi.create_predictions_df(test_df, best_th)
    print(f"predictions.head()={predictions.head()}")
  
    # Fix missing predictions
    # From https://www.kaggle.com/code/jpbremer/backfins-arcface-tpu-effnet/notebook
    public_predictions = pd.read_csv(PUBLIC_SUBMISSION_CSV_PATH)
    ids_without_backfin = np.load(IDS_WITHOUT_BACKFIN_PATH, allow_pickle=True)

    ids2 = public_predictions["image"][~public_predictions["image"].isin(predictions["image"])]

    predictions = pd.concat(
        [
            predictions[~(predictions["image"].isin(ids_without_backfin))],
            public_predictions[public_predictions["image"].isin(ids_without_backfin)],
            public_predictions[public_predictions["image"].isin(ids2)],
        ]
    )
    predictions = predictions.drop_duplicates()

    predictions.to_csv(SUBMISSION_CSV_PATH, index=False)

    end = time.time()
    time_elapsed = end - start
    print('Ensembling complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print ("submmisson.csv generated! 💯🦀🦀")

    logger.info('Ensembling complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    logger.info("submmisson.csv generated! 💯🦀🦀")

###Main Entry
if __name__=='__main__':
    set_seed(CONFIG['seed'])
    print (f"parameter_number=={len(sys.argv)}")

    if len(sys.argv) <= 2:
            print(f"Parameters error!!!!\nIt should be: '{sys.argv[0]} T(t) val_num' / '{sys.argv[0]} I(i) val_num' to use 'Training' or 'Inference' model.")
            sys.exit()
    elif sys.argv[1].lower() != "t" and sys.argv[1].lower() != "i" and sys.argv[1].lower() != "e":
            print(f"Parameters error!!!!\nIt should be: '{sys.argv[0]} T(t) val_num' / '{sys.argv[0]} I(i) val_num' to use 'Training' or 'Inference' model.")
            sys.exit()
    else:
        val_fold_input_params = int(sys.argv[2])

        if sys.argv[1].lower() == "t":
            ##Train DataFrame
            train_df = pd.read_csv(TRAIN_CSV_PATH)
            train_df["image_path"] = train_df["image"].apply(get_image_path, dir=TRAIN_DIR)

            encoder = LabelEncoder()
            train_df["individual_id"] = encoder.fit_transform(train_df["individual_id"])
            np.save(ENCODER_CLASSES_PATH, encoder.classes_)

            skf = StratifiedKFold(n_splits=N_SPLITS)
            for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.individual_id)):
                train_df.loc[val_, "kfold"] = fold   
            train_df.to_csv(TRAIN_CSV_ENCODED_FOLDED_PATH, index=False)

            ##Test DataFrame
            # Use sample submission csv as template
            test_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
            test_df["image_path"] = test_df["image"].apply(get_image_path, dir=TEST_DIR)
            test_df.drop(columns=["predictions"], inplace=True)

            # Dummy id
            test_df["individual_id"] = 0
            test_df.to_csv(TEST_CSV_PATH, index=False)

            Training_and_Validation(mode=0, val_fold=val_fold_input_params)

            """
            CONFIG["model_name"] = "tf_efficientnetv2_xl_in21k"
            CONFIG["model_suffix"] = "effv2XL"
            CONFIG.update({"train_batch_size": 10, "valid_batch_size": 10})
            Training_and_Validation(mode=0, val_fold=val_fold_input_params)
            """

        elif sys.argv[1].lower() == "i":
            Inference(val_fold=val_fold_input_params, image_size=CONFIG['train_img_size'])
        elif sys.argv[1].lower() == "e":
            #EnsembleingGroups(val_fold=val_fold_input_params, image_size=CONFIG['train_img_size'])
            EnsembleingGroups2(val_fold=val_fold_input_params, image_size=CONFIG['train_img_size'])








       
        