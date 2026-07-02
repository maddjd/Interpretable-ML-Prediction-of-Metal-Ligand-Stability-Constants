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
from mindspore.train.callback import Callback, LossMonitor

warnings.filterwarnings("ignore")


try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

FEATURE_FILE = os.path.join(SCRIPT_DIR, "Dataset_features.csv")
TARGET_META_FILE = os.path.join(SCRIPT_DIR, "Dataset_targets.csv")
TARGET_COLUMN = "logK"

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "CNN_Final_Results")

TEST_SIZE = 0.20
RANDOM_STATE = 10
N_SPLITS = 5
CV_RANDOM_STATE = 10
MODEL_SEED = 42

MISSING_THRESHOLD = 0.60
CORRELATION_THRESHOLD = 0.9
CLIP_LOWER = -10.0
CLIP_UPPER = 10.0

best_params = {
    "learning_rate":0.000422013836916548,
    "dropout_rate": 0.0667139556613772,
    "weight_decay": 0.000882104618090643,
    "batch_size": 32,
    "out_channels": 128,
    "kernel_size": 3,
    "fc1_units": 94,
}

NUM_EPOCHS = 1000
AD_RESIDUAL_THRESHOLD = 3.0


TRAIN_CSV = "CNN_Train_Results.csv"
TEST_CSV = "CNN_Test_Results.csv"
AD_CSV = "CNN_AD_Data.csv"
EXPECTED_FILES = (
    TRAIN_CSV,
    TEST_CSV,
    AD_CSV,
)


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


def initialize_output_directory():
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


def read_csv_flexible(path: str) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "gb18030", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Unable to read file: {path}\nLast error: {last_error}")


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "RMSE": float(sqrt(mse)),
    }



class FoldPreprocessor:

    def __init__(
        self,
        missing_threshold=0.60,
        correlation_threshold=0.9,
        clip_lower=-10.0,
        clip_upper=10.0,
    ):
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

        self.selected_columns = []
        self.medians = None
        self.scaler = None

    @staticmethod
    def _numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result = result.replace([np.inf, -np.inf], np.nan)
        for column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        return result

    def fit(self, x_train_df: pd.DataFrame):
        numeric = self._numeric_dataframe(x_train_df)


        keep_columns = numeric.columns[
            numeric.isna().mean() <= self.missing_threshold
        ].tolist()
        numeric = numeric[keep_columns]
        if numeric.shape[1] == 0:
            raise ValueError("No features remain after missing-value filtering.")


        unique_counts = numeric.nunique(dropna=True)
        numeric = numeric.loc[:, unique_counts > 1]
        if numeric.shape[1] == 0:
            raise ValueError("No features remain after constant-feature removal.")


        medians_all = numeric.median(numeric_only=True)
        numeric_imputed = numeric.fillna(medians_all)


        corr = numeric_imputed.corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape, dtype=bool), k=1)
        )
        correlated_to_drop = [
            column
            for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]
        numeric_imputed = numeric_imputed.drop(
            columns=correlated_to_drop,
            errors="ignore",
        )
        if numeric_imputed.shape[1] == 0:
            raise ValueError("No features remain after correlation filtering.")

        self.selected_columns = numeric_imputed.columns.tolist()
        self.medians = medians_all[self.selected_columns]


        self.scaler = RobustScaler()
        self.scaler.fit(numeric_imputed[self.selected_columns])
        return self

    def transform(self, x_df: pd.DataFrame) -> np.ndarray:
        numeric = self._numeric_dataframe(x_df)
        aligned = pd.DataFrame(index=numeric.index)

        for column in self.selected_columns:
            if column in numeric.columns:
                aligned[column] = numeric[column]
            else:
                aligned[column] = np.nan

        aligned = aligned.fillna(self.medians)
        transformed = self.scaler.transform(aligned[self.selected_columns])
        transformed = np.clip(
            transformed,
            self.clip_lower,
            self.clip_upper,
        ).astype(np.float32)

        if not np.all(np.isfinite(transformed)):
            raise ValueError("NaN or infinite values remain after preprocessing.")
        return transformed

    def fit_transform(self, x_train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(x_train_df).transform(x_train_df)


class CNN(nn.Cell):
    def __init__(
        self,
        input_width,
        out_channels,
        kernel_size,
        fc1_units,
        dropout_rate,
    ):
        super(CNN, self).__init__()

        if input_width < kernel_size:
            raise ValueError(
                f"The final feature count ({input_width}) is smaller than the kernel size ({kernel_size})."
            )

        self.conv1 = nn.Conv1d(
            1,
            out_channels,
            kernel_size=kernel_size,
            pad_mode="valid",
            has_bias=True,
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1_input_size = out_channels * (input_width - kernel_size + 1)
        self.fc1 = nn.Dense(self.fc1_input_size, fc1_units)
        self.fc2 = nn.Dense(fc1_units, 1)
        self.dropout = nn.Dropout(keep_prob=1.0 - dropout_rate)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LossRecorderWithEvaluation(Callback):

    def __init__(self, model_ms, x_eval, y_eval):
        super().__init__()
        self.model_ms = model_ms
        self.x_eval = Tensor(np.asarray(x_eval, dtype=np.float32), ms.float32)
        self.y_eval = np.asarray(y_eval, dtype=np.float32).reshape(-1)
        self.train_losses = []
        self.eval_losses = []
        self._step_losses = []

    @staticmethod
    def _to_float(value):
        try:
            array = value.asnumpy()
        except AttributeError:
            array = value
        return float(np.asarray(array).reshape(-1)[0])

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        self._step_losses.append(self._to_float(cb_params.net_outputs))

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        current_epoch = int(cb_params.cur_epoch_num)

        if self._step_losses:
            train_loss = float(np.mean(self._step_losses))
        else:
            train_loss = self._to_float(cb_params.net_outputs)
        self._step_losses = []
        self.train_losses.append(train_loss)

        eval_prediction = (
            self.model_ms.predict(self.x_eval).asnumpy().reshape(-1)
        )
        eval_loss = float(mean_squared_error(self.y_eval, eval_prediction))
        self.eval_losses.append(eval_loss)

        if current_epoch == 1 or current_epoch % 20 == 0:
            print(
                f"Epoch {current_epoch:4d}/{NUM_EPOCHS} | "
                f"train MSE={train_loss:.8f} | "
                f"internal validation MSE={eval_loss:.8f}"
            )


class TrainingLossRecorder(Callback):

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self._step_losses = []

    @staticmethod
    def _to_float(value):
        try:
            array = value.asnumpy()
        except AttributeError:
            array = value
        return float(np.asarray(array).reshape(-1)[0])

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        self._step_losses.append(self._to_float(cb_params.net_outputs))

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        current_epoch = int(cb_params.cur_epoch_num)

        if self._step_losses:
            train_loss = float(np.mean(self._step_losses))
        else:
            train_loss = self._to_float(cb_params.net_outputs)
        self._step_losses = []
        self.train_losses.append(train_loss)

        if current_epoch == 1 or current_epoch % 20 == 0:
            print(
                f"Epoch {current_epoch:4d}/{NUM_EPOCHS} | "
                f"final training MSE={train_loss:.8f}"
            )


def to_cnn_shape(x_2d: np.ndarray) -> np.ndarray:
    return np.expand_dims(
        np.asarray(x_2d, dtype=np.float32),
        axis=1,
    )


def create_cnn_model(input_width: int):
    network = CNN(
        input_width=input_width,
        out_channels=best_params["out_channels"],
        kernel_size=best_params["kernel_size"],
        fc1_units=best_params["fc1_units"],
        dropout_rate=best_params["dropout_rate"],
    )

    optimizer = Adam(
        params=network.trainable_params(),
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )

    model_ms = Model(
        network=network,
        loss_fn=nn.MSELoss(),
        optimizer=optimizer,
    )
    return network, model_ms


def train_internal_fold_model(x_train, y_train, x_valid, y_valid, seed):
    set_all_seeds(seed)

    x_train_cnn = to_cnn_shape(x_train)
    x_valid_cnn = to_cnn_shape(x_valid)
    y_train_2d = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

    network, model_ms = create_cnn_model(input_width=x_train.shape[1])

    train_dataset = NumpySlicesDataset(
        (x_train_cnn, y_train_2d),
        shuffle=True,
    ).batch(
        best_params["batch_size"],
        drop_remainder=False,
    )

    recorder = LossRecorderWithEvaluation(
        model_ms=model_ms,
        x_eval=x_valid_cnn,
        y_eval=y_valid,
    )

    model_ms.train(
        NUM_EPOCHS,
        train_dataset,
        callbacks=[
            LossMonitor(per_print_times=max(1, train_dataset.get_dataset_size())),
            recorder,
        ],
        dataset_sink_mode=False,
    )

    network.set_train(False)
    train_prediction = (
        model_ms.predict(Tensor(x_train_cnn, ms.float32))
        .asnumpy()
        .reshape(-1)
    )
    valid_prediction = (
        model_ms.predict(Tensor(x_valid_cnn, ms.float32))
        .asnumpy()
        .reshape(-1)
    )

    return (
        train_prediction,
        valid_prediction,
        recorder.train_losses,
        recorder.eval_losses,
    )


def train_final_model(x_train, y_train, x_test, seed):
    set_all_seeds(seed)

    x_train_cnn = to_cnn_shape(x_train)
    x_test_cnn = to_cnn_shape(x_test)
    y_train_2d = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

    network, model_ms = create_cnn_model(input_width=x_train.shape[1])

    train_dataset = NumpySlicesDataset(
        (x_train_cnn, y_train_2d),
        shuffle=True,
    ).batch(
        best_params["batch_size"],
        drop_remainder=False,
    )

    recorder = TrainingLossRecorder()

    model_ms.train(
        NUM_EPOCHS,
        train_dataset,
        callbacks=[
            LossMonitor(per_print_times=max(1, train_dataset.get_dataset_size())),
            recorder,
        ],
        dataset_sink_mode=False,
    )

    network.set_train(False)
    train_prediction = (
        model_ms.predict(Tensor(x_train_cnn, ms.float32))
        .asnumpy()
        .reshape(-1)
    )
    test_prediction = (
        model_ms.predict(Tensor(x_test_cnn, ms.float32))
        .asnumpy()
        .reshape(-1)
    )

    return train_prediction, test_prediction, recorder.train_losses


def build_prediction_results(
    filtered_indices,
    original_indices,
    y_true,
    y_pred,
    split_name,
):
    metrics = regression_metrics(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    residual = y_true - y_pred

    return pd.DataFrame({
        "Split": split_name,
        "Filtered_Row_Index": np.asarray(filtered_indices, dtype=int),
        "Original_Row_Index": np.asarray(original_indices, dtype=int),
        "Experimental_logK": y_true,
        "Predicted_logK": y_pred,
        "Residual_True_minus_Predicted": residual,
        "Absolute_Error": np.abs(residual),
        "Squared_Error": residual ** 2,
        "Dataset_R2": metrics["R2"],
        "Dataset_MAE": metrics["MAE"],
        "Dataset_MSE": metrics["MSE"],
        "Dataset_RMSE": metrics["RMSE"],
    })


def calculate_applicability_domain(
    x_train,
    x_test,
    train_indices,
    test_indices,
    original_row_indices,
    y_train,
    train_prediction,
    y_test,
    test_prediction,
):
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)
    train_prediction = np.asarray(train_prediction, dtype=float).reshape(-1)
    test_prediction = np.asarray(test_prediction, dtype=float).reshape(-1)

    train_residual = y_train - train_prediction
    test_residual = y_test - test_prediction

    train_residual_std = float(np.std(train_residual))
    if train_residual_std < 1e-12:
        train_residual_std = 1e-12

    train_standardized_residual = train_residual / train_residual_std
    test_standardized_residual = test_residual / train_residual_std

    train_average_distance = np.mean(
        cdist(x_train, x_train, metric="euclidean"),
        axis=1,
    )
    test_average_distance = np.mean(
        cdist(x_test, x_train, metric="euclidean"),
        axis=1,
    )

    train_outside = (
        np.abs(train_standardized_residual) > AD_RESIDUAL_THRESHOLD
    )
    test_outside = (
        np.abs(test_standardized_residual) > AD_RESIDUAL_THRESHOLD
    )

    train_df = pd.DataFrame({
        "Split": "Training",
        "Filtered_Row_Index": np.asarray(train_indices, dtype=int),
        "Original_Row_Index": original_row_indices[train_indices],
        "Experimental_logK": y_train,
        "Predicted_logK": train_prediction,
        "Residual_True_minus_Predicted": train_residual,
        "Average_Feature_Distance": train_average_distance,
        "Standardized_Residual": train_standardized_residual,
        "AD_Residual_Threshold": AD_RESIDUAL_THRESHOLD,
        "Outside_Applicability_Domain": train_outside,
        "Outside_Direction": np.where(
            train_standardized_residual > AD_RESIDUAL_THRESHOLD,
            "Above_+3",
            np.where(
                train_standardized_residual < -AD_RESIDUAL_THRESHOLD,
                "Below_-3",
                "Inside",
            ),
        ),
    })

    test_df = pd.DataFrame({
        "Split": "Testing",
        "Filtered_Row_Index": np.asarray(test_indices, dtype=int),
        "Original_Row_Index": original_row_indices[test_indices],
        "Experimental_logK": y_test,
        "Predicted_logK": test_prediction,
        "Residual_True_minus_Predicted": test_residual,
        "Average_Feature_Distance": test_average_distance,
        "Standardized_Residual": test_standardized_residual,
        "AD_Residual_Threshold": AD_RESIDUAL_THRESHOLD,
        "Outside_Applicability_Domain": test_outside,
        "Outside_Direction": np.where(
            test_standardized_residual > AD_RESIDUAL_THRESHOLD,
            "Above_+3",
            np.where(
                test_standardized_residual < -AD_RESIDUAL_THRESHOLD,
                "Below_-3",
                "Inside",
            ),
        ),
    })

    ad_df = pd.concat([train_df, test_df], ignore_index=True)
    ad_df["Training_Residual_Standard_Deviation"] = train_residual_std
    return ad_df, train_residual_std


def main():
    initialize_output_directory()
    print("Loading data...")

    feature_raw = read_csv_flexible(FEATURE_FILE)
    meta_raw = read_csv_flexible(TARGET_META_FILE)

    if TARGET_COLUMN not in meta_raw.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' was not found.")

    if len(feature_raw) != len(meta_raw):
        raise ValueError(
            f"The feature file contains {len(feature_raw)} rows, "
            f"whereas the target file contains {len(meta_raw)} rows."
        )

    y = pd.to_numeric(
        meta_raw[TARGET_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)
    original_row_indices = np.arange(len(y), dtype=int)
    valid_target_mask = np.isfinite(y)

    if not np.all(valid_target_mask):
        n_invalid = int(np.sum(~valid_target_mask))
        print(f"Removed samples with missing or nonnumeric targets: {n_invalid}")
        feature_raw = feature_raw.loc[
            valid_target_mask
        ].reset_index(drop=True)
        y = y[valid_target_mask]
        original_row_indices = original_row_indices[valid_target_mask]

    feature_df = feature_raw.copy()
    if feature_df.shape[1] == 0:
        raise ValueError(
            "No candidate features were found in the feature file."
        )

    all_indices = np.arange(len(y), dtype=int)
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    print(f"Valid samples: {len(y)}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Independent test samples: {len(test_indices)}")
    print("Five-fold cross-validation is performed within the training set.")
    print(f"Each fold model and the final model are trained for {NUM_EPOCHS} epochs.")


    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    oof_prediction = np.full(len(train_indices), np.nan, dtype=float)
    fold_training_metrics = []
    fold_validation_metrics = []

    for fold_number, (fold_train_local, fold_valid_local) in enumerate(
        kfold.split(train_indices),
        start=1,
    ):
        print("\n" + "=" * 78)
        print(f"Fold {fold_number}/{N_SPLITS}")

        fold_train_indices = train_indices[fold_train_local]
        fold_valid_indices = train_indices[fold_valid_local]

        preprocessor = FoldPreprocessor(
            missing_threshold=MISSING_THRESHOLD,
            correlation_threshold=CORRELATION_THRESHOLD,
            clip_lower=CLIP_LOWER,
            clip_upper=CLIP_UPPER,
        )
        x_fold_train = preprocessor.fit_transform(
            feature_df.iloc[fold_train_indices]
        )
        x_fold_valid = preprocessor.transform(
            feature_df.iloc[fold_valid_indices]
        )

        print(f"Input features in this fold: {x_fold_train.shape[1]}")

        (
            fold_train_prediction,
            fold_valid_prediction,
            fold_train_losses,
            fold_valid_losses,
        ) = train_internal_fold_model(
            x_train=x_fold_train,
            y_train=y[fold_train_indices],
            x_valid=x_fold_valid,
            y_valid=y[fold_valid_indices],
            seed=MODEL_SEED + fold_number,
        )

        train_metrics_fold = regression_metrics(
            y[fold_train_indices],
            fold_train_prediction,
        )
        validation_metrics_fold = regression_metrics(
            y[fold_valid_indices],
            fold_valid_prediction,
        )

        fold_training_metrics.append(train_metrics_fold)
        fold_validation_metrics.append(validation_metrics_fold)
        oof_prediction[fold_valid_local] = fold_valid_prediction

        print(
            f"Fold {fold_number} training subset: "
            f"R²={train_metrics_fold['R2']:.6f}, "
            f"MAE={train_metrics_fold['MAE']:.6f}, "
            f"MSE={train_metrics_fold['MSE']:.6f}, "
            f"RMSE={train_metrics_fold['RMSE']:.6f}"
        )
        print(
            f"Fold {fold_number} validation subset: "
            f"R²={validation_metrics_fold['R2']:.6f}, "
            f"MAE={validation_metrics_fold['MAE']:.6f}, "
            f"MSE={validation_metrics_fold['MSE']:.6f}, "
            f"RMSE={validation_metrics_fold['RMSE']:.6f}"
        )

    if not np.all(np.isfinite(oof_prediction)):
        raise RuntimeError("Some training samples did not receive out-of-fold predictions.")

    cv_oof_metrics = regression_metrics(
        y[train_indices],
        oof_prediction,
    )


    print("\n" + "=" * 78)
    print("Training the final CNN model on the full 80% training set...")

    final_preprocessor = FoldPreprocessor(
        missing_threshold=MISSING_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
        clip_lower=CLIP_LOWER,
        clip_upper=CLIP_UPPER,
    )
    x_train = final_preprocessor.fit_transform(
        feature_df.iloc[train_indices]
    )
    x_test = final_preprocessor.transform(
        feature_df.iloc[test_indices]
    )

    print(f"Final model input features: {x_train.shape[1]}")

    train_prediction, test_prediction, _ = train_final_model(
        x_train=x_train,
        y_train=y[train_indices],
        x_test=x_test,
        seed=MODEL_SEED,
    )

    train_metrics = regression_metrics(
        y[train_indices],
        train_prediction,
    )
    test_metrics = regression_metrics(
        y[test_indices],
        test_prediction,
    )


    train_results = build_prediction_results(
        filtered_indices=train_indices,
        original_indices=original_row_indices[train_indices],
        y_true=y[train_indices],
        y_pred=train_prediction,
        split_name="Training",
    )
    test_results = build_prediction_results(
        filtered_indices=test_indices,
        original_indices=original_row_indices[test_indices],
        y_true=y[test_indices],
        y_pred=test_prediction,
        split_name="Testing",
    )

    train_results.to_csv(
        os.path.join(OUTPUT_DIR, TRAIN_CSV),
        index=False,
        encoding="utf-8-sig",
    )
    test_results.to_csv(
        os.path.join(OUTPUT_DIR, TEST_CSV),
        index=False,
        encoding="utf-8-sig",
    )

    ad_df, train_residual_std = calculate_applicability_domain(
        x_train=x_train,
        x_test=x_test,
        train_indices=train_indices,
        test_indices=test_indices,
        original_row_indices=original_row_indices,
        y_train=y[train_indices],
        train_prediction=train_prediction,
        y_test=y[test_indices],
        test_prediction=test_prediction,
    )
    ad_df.to_csv(
        os.path.join(OUTPUT_DIR, AD_CSV),
        index=False,
        encoding="utf-8-sig",
    )


    train_ad = ad_df[ad_df["Split"] == "Training"]
    test_ad = ad_df[ad_df["Split"] == "Testing"]
    train_outside_count = int(
        train_ad["Outside_Applicability_Domain"].sum()
    )
    test_outside_count = int(
        test_ad["Outside_Applicability_Domain"].sum()
    )


    validation_r2 = np.asarray(
        [m["R2"] for m in fold_validation_metrics],
        dtype=float,
    )
    validation_mae = np.asarray(
        [m["MAE"] for m in fold_validation_metrics],
        dtype=float,
    )
    validation_mse = np.asarray(
        [m["MSE"] for m in fold_validation_metrics],
        dtype=float,
    )
    validation_rmse = np.asarray(
        [m["RMSE"] for m in fold_validation_metrics],
        dtype=float,
    )

    train_above = int(
        (train_ad["Standardized_Residual"] > AD_RESIDUAL_THRESHOLD).sum()
    )
    train_below = int(
        (train_ad["Standardized_Residual"] < -AD_RESIDUAL_THRESHOLD).sum()
    )
    test_above = int(
        (test_ad["Standardized_Residual"] > AD_RESIDUAL_THRESHOLD).sum()
    )
    test_below = int(
        (test_ad["Standardized_Residual"] < -AD_RESIDUAL_THRESHOLD).sum()
    )

    print("\n" + "=" * 78)
    print("CNN training and evaluation completed")
    print(
        "Mean five-fold validation metrics: "
        f"R²={validation_r2.mean():.6f} ± "
        f"{validation_r2.std(ddof=1):.6f}, "
        f"MAE={validation_mae.mean():.6f} ± "
        f"{validation_mae.std(ddof=1):.6f}, "
        f"MSE={validation_mse.mean():.6f} ± "
        f"{validation_mse.std(ddof=1):.6f}, "
        f"RMSE={validation_rmse.mean():.6f} ± "
        f"{validation_rmse.std(ddof=1):.6f}"
    )
    print(
        "Overall five-fold OOF performance: "
        f"R²={cv_oof_metrics['R2']:.6f}, "
        f"MAE={cv_oof_metrics['MAE']:.6f}, "
        f"MSE={cv_oof_metrics['MSE']:.6f}, "
        f"RMSE={cv_oof_metrics['RMSE']:.6f}"
    )
    print(
        "Final training set: "
        f"R²={train_metrics['R2']:.6f}, "
        f"MAE={train_metrics['MAE']:.6f}, "
        f"MSE={train_metrics['MSE']:.6f}, "
        f"RMSE={train_metrics['RMSE']:.6f}"
    )
    print(
        "Independent test set: "
        f"R²={test_metrics['R2']:.6f}, "
        f"MAE={test_metrics['MAE']:.6f}, "
        f"MSE={test_metrics['MSE']:.6f}, "
        f"RMSE={test_metrics['RMSE']:.6f}"
    )
    print(
        f"Training samples outside the applicability domain: "
        f"{train_outside_count}/{len(train_ad)}; "
        f">+3: {train_above}, <-3: {train_below}"
    )
    print(
        f"Test samples outside the applicability domain: "
        f"{test_outside_count}/{len(test_ad)}; "
        f">+3: {test_above}, <-3: {test_below}"
    )
    print(
        "Training residual standard deviation for applicability-domain analysis: "
        f"{train_residual_std:.8f}"
    )

    actual_files = sorted(
        name
        for name in os.listdir(OUTPUT_DIR)
        if os.path.isfile(os.path.join(OUTPUT_DIR, name))
    )
    expected_files = sorted(EXPECTED_FILES)
    if actual_files != expected_files:
        raise RuntimeError(
            "The number or names of output files do not match the expected set.\n"
            f"Expected: {expected_files}\n"
            f"Actual: {actual_files}"
        )

    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("Generated 3 output files:")
    for filename in actual_files:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
