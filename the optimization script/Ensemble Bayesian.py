# -*- coding: utf-8 -*-

import os
import json
import random
import warnings
from math import sqrt

import numpy as np
import pandas as pd
import optuna

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
FEATURE_FILE = os.path.join(SCRIPT_DIR, "Dataset_features.csv")
TARGET_META_FILE = os.path.join(SCRIPT_DIR, "Dataset_targets.csv")
TARGET_COLUMN = "logK"

OUTPUT_DIR = "Bayesian_Stacking_Core_Output"

TEST_SIZE = 0.2
RANDOM_STATE = 10
N_SPLITS = 5
MODEL_SEED = 42

RF_TRIALS = 50
GBR_TRIALS = 50
RIDGE_TRIALS = 30

AD_THRESHOLD = 3.0


def read_csv(path):
    for e in ["utf-8-sig", "gb18030", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=e)
        except:
            pass
    raise IOError(path)


def metrics(y, p):
    mse = mean_squared_error(y, p)
    return {
        "R2": r2_score(y, p),
        "MAE": mean_absolute_error(y, p),
        "MSE": mse,
        "RMSE": sqrt(mse)
    }


def clean_features(df):
    drop = [
        "logK","LogK","target","Target","SMILES","smiles",
        "ligand_smiles","Ligand_SMILES","ligand",
        "Ligand","ligand_id","Ligand_ID"
    ]
    return df.drop(columns=[x for x in drop if x in df.columns], errors="ignore")


class Preprocessor:
    def fit(self, df):
        df=df.copy()
        num=df.select_dtypes(include=np.number)
        self.cols=[c for c in num.columns if num[c].isna().mean()<0.6 and num[c].nunique()>1]
        self.med=num[self.cols].median()
        self.scaler=RobustScaler().fit(num[self.cols].fillna(self.med))
        return self

    def transform(self, df):
        num=df.reindex(columns=self.cols).fillna(self.med)
        return np.clip(self.scaler.transform(num),-10,10).astype(np.float32)


def rf_model(p):
    return RandomForestRegressor(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        min_samples_leaf=p["min_samples_leaf"],
        max_features=p["max_features"],
        random_state=MODEL_SEED,
        n_jobs=-1
    )


def gbr_model(p):
    return GradientBoostingRegressor(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        min_samples_leaf=p["min_samples_leaf"],
        subsample=p["subsample"],
        random_state=MODEL_SEED
    )


def cv_optimize(build):
    def objective(trial):
        params=build(trial)
        score=[]
        for tr,va in folds:
            m=params["model"](params["params"])
            m.fit(X.iloc[tr], y[tr])
            score.append(r2_score(y[va],m.predict(X.iloc[va])))
        return np.mean(score)
    return objective


feature_raw=read_csv(FEATURE_FILE)
target_raw=read_csv(TARGET_META_FILE)

y=pd.to_numeric(target_raw[TARGET_COLUMN],errors="coerce").values
mask=np.isfinite(y)
feature_raw=feature_raw.loc[mask].reset_index(drop=True)
y=y[mask]

X=clean_features(feature_raw)

idx=np.arange(len(y))
train_idx,test_idx=train_test_split(
    idx,test_size=TEST_SIZE,random_state=RANDOM_STATE
)

X_train_raw=X.iloc[train_idx]
X_test_raw=X.iloc[test_idx]

prep=Preprocessor().fit(X_train_raw)
X_train=pd.DataFrame(prep.transform(X_train_raw))
X_test=pd.DataFrame(prep.transform(X_test_raw))

kf=KFold(N_SPLITS,shuffle=True,random_state=RANDOM_STATE)
folds=list(kf.split(X_train))


def rf_space(trial):
    return {
        "params":{
            "n_estimators":trial.suggest_int("n_estimators",100,800),
            "max_depth":trial.suggest_int("max_depth",5,30),
            "min_samples_split":trial.suggest_int("min_samples_split",2,10),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,5),
            "max_features":trial.suggest_categorical("max_features",["sqrt","log2"])
        },
        "model":rf_model
    }


def gbr_space(trial):
    return {
        "params":{
            "n_estimators":trial.suggest_int("n_estimators",100,800),
            "learning_rate":trial.suggest_float("learning_rate",0.001,0.2),
            "max_depth":trial.suggest_int("max_depth",1,6),
            "min_samples_split":trial.suggest_int("min_samples_split",2,20),
            "min_samples_leaf":trial.suggest_int("min_samples_leaf",1,10),
            "subsample":trial.suggest_float("subsample",0.5,1)
        },
        "model":gbr_model
    }


rf_study=optuna.create_study(direction="maximize")
rf_study.optimize(cv_optimize(rf_space),n_trials=RF_TRIALS)

gbr_study=optuna.create_study(direction="maximize")
gbr_study.optimize(cv_optimize(gbr_space),n_trials=GBR_TRIALS)

rf_params=rf_study.best_params
gbr_params=gbr_study.best_params

oof=np.zeros((len(train_idx),2))

for tr,va in folds:
    rf=rf_model(rf_params)
    gb=gbr_model(gbr_params)
    rf.fit(X_train.iloc[tr],y[train_idx][tr])
    gb.fit(X_train.iloc[tr],y[train_idx][tr])
    oof[va,0]=rf.predict(X_train.iloc[va])
    oof[va,1]=gb.predict(X_train.iloc[va])


def ridge_obj(trial):
    alpha=trial.suggest_float("alpha",1e-6,1000,log=True)
    s=[]
    for tr,va in folds:
        m=Ridge(alpha=alpha)
        m.fit(oof[tr],y[train_idx][tr])
        s.append(r2_score(y[train_idx][va],m.predict(oof[va])))
    return np.mean(s)


ridge_study=optuna.create_study(direction="maximize")
ridge_study.optimize(ridge_obj,n_trials=RIDGE_TRIALS)

ridge=Ridge(alpha=ridge_study.best_params["alpha"])
ridge.fit(oof,y[train_idx])

rf=rf_model(rf_params)
gb=gbr_model(gbr_params)

rf.fit(X_train,y[train_idx])
gb.fit(X_train,y[train_idx])

train_base=np.column_stack([rf.predict(X_train),gb.predict(X_train)])
test_base=np.column_stack([rf.predict(X_test),gb.predict(X_test)])

train_pred=ridge.predict(train_base)
test_pred=ridge.predict(test_base)

train_res=y[train_idx]-train_pred
test_res=y[test_idx]-test_pred

dist_train=np.mean(cdist(X_train,X_train),axis=1)
dist_test=np.mean(cdist(X_test,X_train),axis=1)

train_out=pd.DataFrame({
"split":"train",
"true_logK":y[train_idx],
"predicted_logK":train_pred,
"residual":train_res,
"distance":dist_train,
"outside_AD":abs(train_res/np.std(train_res))>AD_THRESHOLD
})

test_out=pd.DataFrame({
"split":"test",
"true_logK":y[test_idx],
"predicted_logK":test_pred,
"residual":test_res,
"distance":dist_test,
"outside_AD":abs(test_res/np.std(train_res))>AD_THRESHOLD
})

os.makedirs(OUTPUT_DIR,exist_ok=True)

train_out.to_csv(os.path.join(OUTPUT_DIR,"Stacking_Train_Results.csv"),index=False)
test_out.to_csv(os.path.join(OUTPUT_DIR,"Stacking_Test_Results.csv"),index=False)
pd.concat([train_out,test_out]).to_csv(os.path.join(OUTPUT_DIR,"Stacking_AD_Data.csv"),index=False)

with open(os.path.join(OUTPUT_DIR,"best_hyperparameters.json"),"w") as f:
    json.dump({
        "RF":rf_params,
        "GBR":gbr_params,
        "Ridge_alpha":ridge.alpha
    },f,indent=2)
