import os
import random
import shutil
import warnings
from math import sqrt

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.dataset import NumpySlicesDataset
from mindspore.nn import Adam
from mindspore.train import Model

warnings.filterwarnings("ignore")

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

FEATURE_FILE = os.path.join(SCRIPT_DIR, "Dataset_features.csv")
TARGET_META_FILE = os.path.join(SCRIPT_DIR, "Dataset_targets.csv")
TARGET_COLUMN = "logK"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "MLP_Final_Results")

TEST_SIZE = 0.20
RANDOM_STATE = 10
N_SPLITS = 5
CV_RANDOM_STATE = 10
MODEL_SEED = 42
NUM_EPOCHS = 1000

AD_RESIDUAL_THRESHOLD = 3.0

HIDDEN_DIMS = [109, 34, 37]
DROPOUT_RATE = 0.044784571132741716
LEARNING_RATE = 0.00024272037465258528
WEIGHT_DECAY = 0.0005268845927135383
BATCH_SIZE = 64

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def read_csv_flexible(path):
    for enc in ("utf-8-sig", "gb18030", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            pass
    raise RuntimeError(path)


def metrics(y, p):
    mse = mean_squared_error(y, p)
    return {
        "R2": float(r2_score(y, p)),
        "MAE": float(mean_absolute_error(y, p)),
        "MSE": float(mse),
        "RMSE": float(sqrt(mse))
    }


class FoldPreprocessor:
    def __init__(self):
        self.columns = []
        self.medians = None
        self.scaler = None

    def fit(self, df):
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.loc[:, df.isna().mean() <= 0.6]
        df = df.loc[:, df.nunique(dropna=True) > 1]

        self.medians = df.median()
        df = df.fillna(self.medians)

        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape),1).astype(bool))
        drop = [c for c in upper.columns if (upper[c] > 0.9).any()]
        df = df.drop(columns=drop, errors="ignore")

        self.columns = df.columns.tolist()
        self.scaler = RobustScaler()
        self.scaler.fit(df[self.columns])
        return self

    def transform(self, df):
        df = df.apply(pd.to_numeric, errors="coerce")
        x = pd.DataFrame(index=df.index)
        for c in self.columns:
            x[c] = df[c] if c in df else np.nan
        x = x.fillna(self.medians)
        return np.clip(self.scaler.transform(x), -10, 10).astype(np.float32)

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class ResidualBlock(nn.Cell):
    def __init__(self, i, o):
        super().__init__()
        self.fc = nn.Dense(i, o)
        self.act = nn.LeakyReLU()
        self.short = nn.Dense(i, o) if i != o else None

    def construct(self, x):
        y = self.act(self.fc(x))
        return y + (self.short(x) if self.short else x)


class ImprovedMLP(nn.Cell):
    def __init__(self, input_dim):
        super().__init__()
        layers=[]
        d=input_dim
        for h in HIDDEN_DIMS:
            layers.append(ResidualBlock(d,h))
            layers.append(nn.Dropout(keep_prob=1-DROPOUT_RATE))
            d=h
        layers.append(nn.Dense(d,1))
        self.net=nn.SequentialCell(layers)

    def construct(self,x):
        return self.net(x)


def train_model(x,y,xtest):
    set_seed(MODEL_SEED)
    net=ImprovedMLP(x.shape[1])
    opt=Adam(net.trainable_params(),
             learning_rate=LEARNING_RATE,
             weight_decay=WEIGHT_DECAY)
    model=Model(net, nn.MSELoss(), opt)

    ds=NumpySlicesDataset(
        (x,y.reshape(-1,1)),
        shuffle=True).batch(BATCH_SIZE)

    model.train(NUM_EPOCHS, ds, dataset_sink_mode=False)

    net.set_train(False)
    return (
        model.predict(Tensor(x,ms.float32)).asnumpy().reshape(-1),
        model.predict(Tensor(xtest,ms.float32)).asnumpy().reshape(-1)
    )


def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    features=read_csv_flexible(FEATURE_FILE)
    target=read_csv_flexible(TARGET_META_FILE)

    y=pd.to_numeric(target[TARGET_COLUMN],errors="coerce").values
    idx=np.arange(len(y))

    train_idx,test_idx=train_test_split(
        idx,test_size=TEST_SIZE,
        random_state=RANDOM_STATE)

    prep=FoldPreprocessor()
    xtrain=prep.fit_transform(features.iloc[train_idx])
    xtest=prep.transform(features.iloc[test_idx])

    train_pred,test_pred=train_model(
        xtrain,y[train_idx].astype(np.float32),
        xtest)

    pd.DataFrame({
        "Experimental_logK":y[train_idx],
        "Predicted_logK":train_pred
    }).to_csv(
        os.path.join(OUTPUT_DIR,"MLP_Train_Results.csv"),
        index=False)

    pd.DataFrame({
        "Experimental_logK":y[test_idx],
        "Predicted_logK":test_pred
    }).to_csv(
        os.path.join(OUTPUT_DIR,"MLP_Test_Results.csv"),
        index=False)

    dist=np.mean(cdist(xtest,xtrain),axis=1)
    pd.DataFrame({
        "Average_Feature_Distance":dist,
        "Residual":y[test_idx]-test_pred
    }).to_csv(
        os.path.join(OUTPUT_DIR,"MLP_AD_Data.csv"),
        index=False)


if __name__=="__main__":
    main()
