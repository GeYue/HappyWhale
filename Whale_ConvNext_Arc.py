#coding=UTF-8

###Imports
import math, os, sys
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
from PIL import Image
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from timm.data.transforms_factory import create_transform
from timm.optim import create_optimizer_v2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

os.environ['TORCH_HOME'] = "./model/"

###Paths & Settings¶
INPUT_DIR = Path("./") / "data3"
OUTPUT_DIR = Path("./") / "output"

DATA_ROOT_DIR = INPUT_DIR / "convert-backfintfrecords" / "happy-whale-and-dolphin-backfin"
TRAIN_DIR = DATA_ROOT_DIR / "train_images"
TEST_DIR = DATA_ROOT_DIR / "test_images"
TRAIN_CSV_PATH = DATA_ROOT_DIR / "train.csv"
SAMPLE_SUBMISSION_CSV_PATH = DATA_ROOT_DIR / "sample_submission.csv"
#PUBLIC_SUBMISSION_CSV_PATH = INPUT_DIR / "0-720-eff-b5-640-rotate" / "submission.csv"
IDS_WITHOUT_BACKFIN_PATH = INPUT_DIR / "ids-without-backfin" / "ids_without_backfin.npy"

N_SPLITS = 5

ENCODER_CLASSES_PATH = OUTPUT_DIR / "encoder_classes.npy"
TEST_CSV_PATH = OUTPUT_DIR / "test.csv"
TRAIN_CSV_ENCODED_FOLDED_PATH = OUTPUT_DIR / "train_encoded_folded.csv"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
SUBMISSION_CSV_PATH = OUTPUT_DIR / "submission.csv"

DEBUG = False

###Prepare DataFrames
def get_image_path(id: str, dir: Path) -> str:
    return f"{dir / id}"

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
    
train_df.head()

##Test DataFrame
# Use sample submission csv as template
test_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
test_df["image_path"] = test_df["image"].apply(get_image_path, dir=TEST_DIR)
test_df.drop(columns=["predictions"], inplace=True)

# Dummy id
test_df["individual_id"] = 0
test_df.to_csv(TEST_CSV_PATH, index=False)

test_df.head()

###DataSets
class HappyWhaleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df
        self.transform = transform

        self.image_names = self.df["image"].values
        self.image_paths = self.df["image_path"].values
        self.targets = self.df["individual_id"].values

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_name = self.image_names[index]
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)

        target = self.targets[index]
        target = torch.tensor(target, dtype=torch.long)

        return {"image_name": image_name, "image": image, "target": target}

    def __len__(self) -> int:
        return len(self.df)

###Lightning DataModule
class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_encoded_folded: str,
        test_csv: str,
        val_fold: float,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_encoded_folded)
        self.test_df = pd.read_csv(test_csv)
        
        self.transform = create_transform(
            input_size=(self.hparams.image_size, self.hparams.image_size),
            crop_pct=1.0,
        )
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Split train df using fold
            train_df = self.train_df[self.train_df.kfold != self.hparams.val_fold].reset_index(drop=True)
            val_df = self.train_df[self.train_df.kfold == self.hparams.val_fold].reset_index(drop=True)

            self.train_dataset = HappyWhaleDataset(train_df, transform=self.transform)
            self.val_dataset = HappyWhaleDataset(val_df, transform=self.transform)

        if stage == "test" or stage is None:
            self.test_dataset = HappyWhaleDataset(self.test_df, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: HappyWhaleDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=train,
        )

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

###Lightning Module
class LitModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        drop_rate: float,
        embedding_size: int,
        num_classes: int,
        arc_s: float,
        arc_m: float,
        arc_easy_margin: bool,
        arc_ls_eps: float,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        len_train_dl: int,
        epochs:int
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = timm.create_model(model_name, pretrained=pretrained, drop_rate=drop_rate)
        self.embedding = nn.Linear(self.model.get_classifier().in_features, embedding_size)
        self.model.reset_classifier(num_classes=0, global_pool="avg")

        self.arc = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            s=arc_s,
            m=arc_m,
            easy_margin=arc_easy_margin,
            ls_eps=arc_ls_eps,
        )

        self.loss_fn = F.cross_entropy

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        embeddings = self.embedding(features)

        return embeddings

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt=self.hparams.optimizer,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.hparams.learning_rate,
            steps_per_epoch=self.hparams.len_train_dl,
            epochs=self.hparams.epochs,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def _step(self, batch: Dict[str, torch.Tensor], step: str) -> torch.Tensor:
        images, targets = batch["image"], batch["target"]

        embeddings = self(images)
        outputs = self.arc(embeddings, targets, self.device)

        loss = self.loss_fn(outputs, targets)
        
        self.log(f"{step}_loss", loss, batch_size=len(batch))

        return loss

###Training
def train(
    train_csv_encoded_folded: str = str(TRAIN_CSV_ENCODED_FOLDED_PATH),
    test_csv: str = str(TEST_CSV_PATH),
    val_fold: float = 0.0,
    image_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 24,
    model_name: str = "tf_efficientnet_b0",
    pretrained: bool = True,
    drop_rate: float = 0.0,
    embedding_size: int = 512,
    num_classes: int = 15587,
    arc_s: float = 30.0,
    arc_m: float = 0.5,
    arc_easy_margin: bool = False,
    arc_ls_eps: float = 0.0,
    optimizer: str = "adam",
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-6,
    checkpoints_dir: str = str(CHECKPOINTS_DIR),
    accumulate_grad_batches: int = 1,
    auto_lr_find: bool = False,
    auto_scale_batch_size: bool = False,
    fast_dev_run: bool = False,
    gpus: int = 1,
    max_epochs: int = 2,
    precision: int = 16,
    stochastic_weight_avg: bool = True):

	pl.seed_everything(42)
	datamodule = LitDataModule(
        train_csv_encoded_folded=train_csv_encoded_folded,
        test_csv=test_csv,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers)

	datamodule.setup()
	len_train_dl = len(datamodule.train_dataloader())

	module = LitModule(
        model_name=model_name,
        pretrained=pretrained,
        drop_rate=drop_rate,
        embedding_size=embedding_size,
        num_classes=num_classes,
        arc_s=arc_s,
        arc_m=arc_m,
        arc_easy_margin=arc_easy_margin,
        arc_ls_eps=arc_ls_eps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        len_train_dl=len_train_dl,
        epochs=max_epochs
    )
    
	model_checkpoint = ModelCheckpoint(
        checkpoints_dir,
        filename=f"{model_name}_{image_size}",
        monitor="val_loss",
    )
        
	trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=auto_scale_batch_size,
        benchmark=True,
        callbacks=[model_checkpoint],
        deterministic=True,
        fast_dev_run=fast_dev_run,
        gpus=gpus,
        max_epochs=2 if DEBUG else max_epochs,
        precision=precision,
        stochastic_weight_avg=stochastic_weight_avg,
        limit_train_batches=0.1 if DEBUG else 1.0,
        limit_val_batches=0.1 if DEBUG else 1.0,
    )

	trainer.tune(module, datamodule=datamodule)

	trainer.fit(module, datamodule=datamodule)


model_name = "convnext_large"
image_size = 64
batch_size = 16

import warnings
warnings.filterwarnings('ignore')

train(model_name=model_name,
      image_size=image_size,
      batch_size=batch_size)


####Inference Part
def load_eval_module(checkpoint_path: str, device: torch.device) -> LitModule:
    module = LitModule.load_from_checkpoint(checkpoint_path)
    module.to(device)
    module.eval()

    return module

def load_dataloaders(
    train_csv_encoded_folded: str,
    test_csv: str,
    val_fold: float,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    datamodule = LitDataModule(
        train_csv_encoded_folded=train_csv_encoded_folded,
        test_csv=test_csv,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    datamodule.setup()

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    return train_dl, val_dl, test_dl


def load_encoder() -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.classes_ = np.load(ENCODER_CLASSES_PATH, allow_pickle=True)

    return encoder


#@torch.inference_mode()
def get_embeddings(
    module: pl.LightningModule, dataloader: DataLoader, encoder: LabelEncoder, stage: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_image_names = []
    all_embeddings = []
    all_targets = []

    for batch in tqdm(dataloader, desc=f"Creating {stage} embeddings"):
        image_names = batch["image_name"]
        images = batch["image"].to(module.device)
        targets = batch["target"].to(module.device)

        embeddings = module(images)

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


def create_and_search_index(embedding_size: int, train_embeddings: np.ndarray, val_embeddings: np.ndarray, k: int):
    index = faiss.IndexFlatIP(embedding_size)
    index.add(train_embeddings)
    D, I = index.search(val_embeddings, k=k)  # noqa: E741

    return D, I


def create_val_targets_df(
    train_targets: np.ndarray, val_image_names: np.ndarray, val_targets: np.ndarray
) -> pd.DataFrame:

    allowed_targets = np.unique(train_targets)
    val_targets_df = pd.DataFrame(np.stack([val_image_names, val_targets], axis=1), columns=["image", "target"])
    val_targets_df.loc[~val_targets_df.target.isin(allowed_targets), "target"] = "new_individual"

    return val_targets_df


def create_distances_df(
    image_names: np.ndarray, targets: np.ndarray, D: np.ndarray, I: np.ndarray, stage: str  # noqa: E741
) -> pd.DataFrame:

    distances_df = []
    for i, image_name in tqdm(enumerate(image_names), desc=f"Creating {stage}_df"):
        target = targets[I[i]]
        distances = D[i]
        subset_preds = pd.DataFrame(np.stack([target, distances], axis=1), columns=["target", "distances"])
        subset_preds["image"] = image_name
        distances_df.append(subset_preds)

    distances_df = pd.concat(distances_df).reset_index(drop=True)
    distances_df = distances_df.groupby(["image", "target"]).distances.max().reset_index()
    distances_df = distances_df.sort_values("distances", ascending=False).reset_index(drop=True)

    return distances_df


def get_best_threshold(val_targets_df: pd.DataFrame, valid_df: pd.DataFrame) -> Tuple[float, float]:
    best_th = 0
    best_cv = 0
    for th in [0.1 * x for x in range(11)]:
        all_preds = get_predictions(valid_df, threshold=th)

        cv = 0
        for i, row in val_targets_df.iterrows():
            target = row.target
            preds = all_preds[row.image]
            val_targets_df.loc[i, th] = map_per_image(target, preds)

        cv = val_targets_df[th].mean()

        print(f"th={th} cv={cv}")

        if cv > best_cv:
            best_th = th
            best_cv = cv

    print(f"best_th={best_th}")
    print(f"best_cv={best_cv}")

    # Adjustment: Since Public lb has nearly 10% 'new_individual' (Be Careful for private LB)
    val_targets_df["is_new_individual"] = val_targets_df.target == "new_individual"
    val_scores = val_targets_df.groupby("is_new_individual").mean().T
    val_scores["adjusted_cv"] = val_scores[True] * 0.1 + val_scores[False] * 0.9
    best_th = val_scores["adjusted_cv"].idxmax()
    print(f"best_th_adjusted={best_th}")

    return best_th, best_cv


def get_predictions(df: pd.DataFrame, threshold: float = 0.2):
    sample_list = ["938b7e931166", "5bf17305f073", "7593d2aee842", "7362d7a01d00", "956562ff2888"]

    predictions = {}
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Creating predictions for threshold={threshold}"):
        if row.image in predictions:
            if len(predictions[row.image]) == 5:
                continue
            predictions[row.image].append(row.target)
        elif row.distances > threshold:
            predictions[row.image] = [row.target, "new_individual"]
        else:
            predictions[row.image] = ["new_individual", row.target]

    for x in tqdm(predictions):
        if len(predictions[x]) < 5:
            remaining = [y for y in sample_list if y not in predictions]
            predictions[x] = predictions[x] + remaining
            predictions[x] = predictions[x][:5]

    return predictions


# TODO: add types
def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def create_predictions_df(test_df: pd.DataFrame, best_th: float) -> pd.DataFrame:
    predictions = get_predictions(test_df, best_th)

    predictions = pd.Series(predictions).reset_index()
    predictions.columns = ["image", "predictions"]
    predictions["predictions"] = predictions["predictions"].apply(lambda x: " ".join(x))

    return predictions

def infer(
    checkpoint_path: str,
    train_csv_encoded_folded: str = str(TRAIN_CSV_ENCODED_FOLDED_PATH),
    test_csv: str = str(TEST_CSV_PATH),
    val_fold: float = 0.0,
    image_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 2,
    k: int = 50,
):
    module = load_eval_module(checkpoint_path, torch.device("cuda"))

    train_dl, val_dl, test_dl = load_dataloaders(
        train_csv_encoded_folded=train_csv_encoded_folded,
        test_csv=test_csv,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    encoder = load_encoder()

    train_image_names, train_embeddings, train_targets = get_embeddings(module, train_dl, encoder, stage="train")
    val_image_names, val_embeddings, val_targets = get_embeddings(module, val_dl, encoder, stage="val")
    test_image_names, test_embeddings, test_targets = get_embeddings(module, test_dl, encoder, stage="test")

    D, I = create_and_search_index(module.hparams.embedding_size, train_embeddings, val_embeddings, k)  # noqa: E741
    print("Created index with train_embeddings")

    val_targets_df = create_val_targets_df(train_targets, val_image_names, val_targets)
    print(f"val_targets_df=\n{val_targets_df.head()}")

    val_df = create_distances_df(val_image_names, train_targets, D, I, "val")
    print(f"val_df=\n{val_df.head()}")

    best_th, best_cv = get_best_threshold(val_targets_df, val_df)
    print(f"val_targets_df=\n{val_targets_df.describe()}")

    train_embeddings = np.concatenate([train_embeddings, val_embeddings])
    train_targets = np.concatenate([train_targets, val_targets])
    print("Updated train_embeddings and train_targets with val data")

    D, I = create_and_search_index(module.hparams.embedding_size, train_embeddings, test_embeddings, k)  # noqa: E741
    print("Created index with train_embeddings")

    test_df = create_distances_df(test_image_names, train_targets, D, I, "test")
    print(f"test_df=\n{test_df.head()}")

    predictions = create_predictions_df(test_df, best_th)
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
    
infer(checkpoint_path=CHECKPOINTS_DIR / f"{model_name}_{image_size}.ckpt", image_size=image_size, batch_size=batch_size)

