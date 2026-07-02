import os
import random
import shutil
import warnings
from math import sqrt

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

FEATURE_FILE = os.path.join(SCRIPT_DIR, "NIST_features.csv")
TARGET_META_FILE = os.path.join(SCRIPT_DIR, "NIST_targets.csv")
TARGET_COLUMN = "logK"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Stacking_Final_Results")

TEST_SIZE = 0.20
RANDOM_STATE = 10
N_SPLITS = 5
CV_RANDOM_STATE = 10
MODEL_SEED = 42

MISSING_THRESHOLD = 0.60
CORRELATION_THRESHOLD = 0.90
MAX_CATEGORY_LEVELS = 40
ROBUST_CLIP_LIMIT = 10.0

BEST_RF_PARAMS = {
    "n_estimators": 416,
    "max_depth": 25,
    "min_samples_split": 7,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
}

BEST_GBR_PARAMS = {
    "n_estimators": 850,
    "learning_rate": 0.16112032072915206,
    "max_depth": 3,
    "min_samples_split": 8,
    "min_samples_leaf": 8,
    "subsample": 0.9954256158008223,
    "max_features": 0.6413087721403323,
    "loss": "squared_error",
}

BEST_RIDGE_ALPHA = 1.0164432062019527e-06
AD_RESIDUAL_THRESHOLD = 3.0

OUTPUT_FILES = {
    "train_csv": "Stacking_Train_Results.csv",
    "test_csv": "Stacking_Test_Results.csv",
    "ad_csv": "Stacking_AD_Data.csv",
}


def initialize_output_directory():
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


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


def format_metrics(metrics):
    return (
        f"R2={metrics['R2']:.6f}, "
        f"MAE={metrics['MAE']:.6f}, "
        f"MSE={metrics['MSE']:.6f}, "
        f"RMSE={metrics['RMSE']:.6f}"
    )



class TrainOnlyPreprocessor:
    def __init__(
        self,
        missing_threshold=0.60,
        correlation_threshold=0.90,
        max_category_levels=40,
        clip_limit=10.0,
    ):
        self.missing_threshold = missing_threshold
        self.correlation_threshold = correlation_threshold
        self.max_category_levels = max_category_levels
        self.clip_limit = clip_limit
        self.numeric_columns = []
        self.categorical_columns = []
        self.numeric_medians = None
        self.scaler = None
        self.encoder = None

    @staticmethod
    def _coerce_numeric_candidates(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy().replace([np.inf, -np.inf], np.nan)
        for column in result.columns:
            if pd.api.types.is_numeric_dtype(result[column]):
                continue
            converted = pd.to_numeric(result[column], errors="coerce")
            non_missing_count = int(result[column].notna().sum())
            if non_missing_count == 0:
                continue
            if converted.notna().sum() / non_missing_count >= 0.95:
                result[column] = converted
        return result

    def fit(self, x_train_df: pd.DataFrame):
        self.numeric_columns = []
        self.categorical_columns = []
        self.numeric_medians = None
        self.scaler = None
        self.encoder = None

        df = self._coerce_numeric_candidates(x_train_df)
        all_numeric_columns = df.select_dtypes(
            include=[np.number, "bool"]
        ).columns.tolist()
        numeric = df[all_numeric_columns].copy()

        if numeric.shape[1] == 0:
            raise ValueError("No usable numeric features were found in the training subset.")

        numeric = numeric.loc[
            :, numeric.isna().mean() <= self.missing_threshold
        ]
        if numeric.shape[1] == 0:
            raise ValueError("No numeric features remain after missing-value filtering.")

        numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
        if numeric.shape[1] == 0:
            raise ValueError("No numeric features remain after constant-feature removal.")

        medians = numeric.median(numeric_only=True)
        numeric_imputed = numeric.fillna(medians)

        correlation_matrix = numeric_imputed.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape, dtype=bool), k=1)
        )
        correlated_to_drop = [
            column
            for column in upper_triangle.columns
            if (upper_triangle[column] > self.correlation_threshold).any()
        ]
        numeric_imputed = numeric_imputed.drop(
            columns=correlated_to_drop,
            errors="ignore",
        )
        if numeric_imputed.shape[1] == 0:
            raise ValueError("No numeric features remain after correlation filtering.")

        self.numeric_columns = numeric_imputed.columns.tolist()
        self.numeric_medians = medians[self.numeric_columns]
        self.scaler = RobustScaler()
        self.scaler.fit(numeric_imputed[self.numeric_columns])

        non_numeric_columns = [
            column for column in df.columns if column not in all_numeric_columns
        ]
        self.categorical_columns = [
            column
            for column in non_numeric_columns
            if 1 < int(df[column].nunique(dropna=True)) <= self.max_category_levels
        ]

        if self.categorical_columns:
            try:
                self.encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                )
            except TypeError:
                self.encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False,
                )
            categorical_fit = (
                df[self.categorical_columns]
                .fillna("__MISSING__")
                .astype(str)
            )
            self.encoder.fit(categorical_fit)

        return self

    def transform(self, x_df: pd.DataFrame) -> np.ndarray:
        df = self._coerce_numeric_candidates(x_df)
        numeric = pd.DataFrame(index=df.index)

        for column in self.numeric_columns:
            if column in df.columns:
                numeric[column] = pd.to_numeric(df[column], errors="coerce")
            else:
                numeric[column] = np.nan

        numeric = numeric.fillna(self.numeric_medians)
        x_numeric = self.scaler.transform(numeric[self.numeric_columns])
        x_numeric = np.clip(
            x_numeric,
            -self.clip_limit,
            self.clip_limit,
        )
        parts = [x_numeric]

        if self.categorical_columns:
            categorical = pd.DataFrame(index=df.index)
            for column in self.categorical_columns:
                if column in df.columns:
                    categorical[column] = df[column]
                else:
                    categorical[column] = "__MISSING__"
            categorical = categorical.fillna("__MISSING__").astype(str)
            parts.append(
                self.encoder.transform(categorical[self.categorical_columns])
            )

        x_out = np.hstack(parts).astype(np.float32)
        if not np.all(np.isfinite(x_out)):
            raise ValueError("NaN or infinite values remain after preprocessing.")
        return x_out

    def fit_transform(self, x_train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(x_train_df).transform(x_train_df)


def build_rf(seed):
    return RandomForestRegressor(
        n_estimators=BEST_RF_PARAMS["n_estimators"],
        max_depth=BEST_RF_PARAMS["max_depth"],
        min_samples_split=BEST_RF_PARAMS["min_samples_split"],
        min_samples_leaf=BEST_RF_PARAMS["min_samples_leaf"],
        max_features=BEST_RF_PARAMS["max_features"],
        bootstrap=BEST_RF_PARAMS["bootstrap"],
        random_state=seed,
        n_jobs=-1,
    )


def build_gbr(seed):
    return GradientBoostingRegressor(
        n_estimators=BEST_GBR_PARAMS["n_estimators"],
        learning_rate=BEST_GBR_PARAMS["learning_rate"],
        max_depth=BEST_GBR_PARAMS["max_depth"],
        min_samples_split=BEST_GBR_PARAMS["min_samples_split"],
        min_samples_leaf=BEST_GBR_PARAMS["min_samples_leaf"],
        subsample=BEST_GBR_PARAMS["subsample"],
        max_features=BEST_GBR_PARAMS["max_features"],
        loss=BEST_GBR_PARAMS["loss"],
        random_state=seed,
    )


def build_meta_model():
    return Ridge(alpha=BEST_RIDGE_ALPHA)


def generate_base_oof(feature_df, y, train_indices):
    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    n_train = len(train_indices)
    oof_base = np.full((n_train, 2), np.nan, dtype=float)
    fold_assignments = np.full(n_train, -1, dtype=int)

    for fold_number, (fold_train_local, fold_valid_local) in enumerate(
        kfold.split(train_indices),
        start=1,
    ):
        print("\n" + "=" * 78)
        print(f"Fold {fold_number}/{N_SPLITS}")

        fold_train_indices = train_indices[fold_train_local]
        fold_valid_indices = train_indices[fold_valid_local]

        preprocessor = TrainOnlyPreprocessor(
            missing_threshold=MISSING_THRESHOLD,
            correlation_threshold=CORRELATION_THRESHOLD,
            max_category_levels=MAX_CATEGORY_LEVELS,
            clip_limit=ROBUST_CLIP_LIMIT,
        )
        x_fold_train = preprocessor.fit_transform(
            feature_df.iloc[fold_train_indices]
        )
        x_fold_valid = preprocessor.transform(
            feature_df.iloc[fold_valid_indices]
        )

        print(
            f"Training samples={len(fold_train_indices)}, "
            f"validation samples={len(fold_valid_indices)}, "
            f"features={x_fold_train.shape[1]}"
        )

        rf_model = build_rf(MODEL_SEED + fold_number)
        gbr_model = build_gbr(MODEL_SEED + fold_number)
        rf_model.fit(x_fold_train, y[fold_train_indices])
        gbr_model.fit(x_fold_train, y[fold_train_indices])

        rf_valid_prediction = rf_model.predict(x_fold_valid)
        gbr_valid_prediction = gbr_model.predict(x_fold_valid)

        oof_base[fold_valid_local, 0] = rf_valid_prediction
        oof_base[fold_valid_local, 1] = gbr_valid_prediction
        fold_assignments[fold_valid_local] = fold_number

        rf_metrics = regression_metrics(
            y[fold_valid_indices],
            rf_valid_prediction,
        )
        gbr_metrics = regression_metrics(
            y[fold_valid_indices],
            gbr_valid_prediction,
        )

        print(f"Fold {fold_number} Random Forest: {format_metrics(rf_metrics)}")
        print(f"Fold {fold_number} Gradient Boosting: {format_metrics(gbr_metrics)}")

    if not np.all(np.isfinite(oof_base)):
        raise RuntimeError("Missing values were found in the base-model OOF predictions.")
    if np.any(fold_assignments < 0):
        raise RuntimeError("Some training samples were not assigned to a validation fold.")

    return oof_base, fold_assignments


def generate_stacking_meta_oof(oof_base, y_train, fold_assignments):
    meta_oof = np.full(len(y_train), np.nan, dtype=float)

    for fold_number in range(1, N_SPLITS + 1):
        valid_mask = fold_assignments == fold_number
        train_mask = ~valid_mask
        meta_model = build_meta_model()
        meta_model.fit(oof_base[train_mask], y_train[train_mask])
        fold_prediction = meta_model.predict(oof_base[valid_mask])
        meta_oof[valid_mask] = fold_prediction
        metrics = regression_metrics(y_train[valid_mask], fold_prediction)
        print(f"Stacking fold {fold_number}: {format_metrics(metrics)}")

    if not np.all(np.isfinite(meta_oof)):
        raise RuntimeError("Missing values were found in the stacking OOF predictions.")
    return meta_oof


def fit_final_stacking_model(
    feature_df,
    y,
    train_indices,
    test_indices,
    oof_base,
):
    final_preprocessor = TrainOnlyPreprocessor(
        missing_threshold=MISSING_THRESHOLD,
        correlation_threshold=CORRELATION_THRESHOLD,
        max_category_levels=MAX_CATEGORY_LEVELS,
        clip_limit=ROBUST_CLIP_LIMIT,
    )

    x_train = final_preprocessor.fit_transform(feature_df.iloc[train_indices])
    x_test = final_preprocessor.transform(feature_df.iloc[test_indices])
    print(f"Final model input features: {x_train.shape[1]}")

    final_rf = build_rf(MODEL_SEED)
    final_gbr = build_gbr(MODEL_SEED)
    final_rf.fit(x_train, y[train_indices])
    final_gbr.fit(x_train, y[train_indices])

    train_base_fullfit = np.column_stack(
        [final_rf.predict(x_train), final_gbr.predict(x_train)]
    )
    test_base = np.column_stack(
        [final_rf.predict(x_test), final_gbr.predict(x_test)]
    )

    final_meta = build_meta_model()
    final_meta.fit(oof_base, y[train_indices])

    return {
        "x_train": x_train,
        "x_test": x_test,
        "fullfit_train_prediction": final_meta.predict(train_base_fullfit),
        "test_prediction": final_meta.predict(test_base),
    }


def build_prediction_results(
    filtered_indices,
    original_indices,
    y_true,
    y_pred,
    split_name,
    prediction_type,
):
    metrics = regression_metrics(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred

    return pd.DataFrame(
        {
            "Split": split_name,
            "Prediction_Type": prediction_type,
            "Filtered_Row_Index": np.asarray(filtered_indices, dtype=int),
            "Original_Row_Index": np.asarray(original_indices, dtype=int),
            "Experimental_logK": y_true,
            "Predicted_logK": y_pred,
            "Residual_True_minus_Predicted": residual,
            "Absolute_Error": np.abs(residual),
            "Squared_Error": residual**2,
            "Dataset_R2": metrics["R2"],
            "Dataset_MAE": metrics["MAE"],
            "Dataset_MSE": metrics["MSE"],
            "Dataset_RMSE": metrics["RMSE"],
        }
    )


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
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    train_prediction = np.asarray(train_prediction, dtype=float)
    test_prediction = np.asarray(test_prediction, dtype=float)

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

    train_df = pd.DataFrame(
        {
            "Split": "Training",
            "Prediction_Type": "Final_FullFit_Training",
            "Filtered_Row_Index": train_indices,
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
            "Training_Residual_Standard_Deviation": train_residual_std,
        }
    )

    test_df = pd.DataFrame(
        {
            "Split": "Testing",
            "Prediction_Type": "Independent_Test",
            "Filtered_Row_Index": test_indices,
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
            "Training_Residual_Standard_Deviation": train_residual_std,
        }
    )

    return pd.concat([train_df, test_df], ignore_index=True)


def main():
    initialize_output_directory()
    set_all_seeds(MODEL_SEED)
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

    y = pd.to_numeric(meta_raw[TARGET_COLUMN], errors="coerce").to_numpy(
        dtype=float
    )
    original_row_indices = np.arange(len(y), dtype=int)
    valid_target_mask = np.isfinite(y)

    if not np.all(valid_target_mask):
        invalid_count = int(np.sum(~valid_target_mask))
        print(f"Removed samples with missing or nonnumeric targets: {invalid_count}")
        feature_raw = feature_raw.loc[valid_target_mask].reset_index(drop=True)
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

    oof_base, fold_assignments = generate_base_oof(
        feature_df=feature_df,
        y=y,
        train_indices=train_indices,
    )

    y_train = y[train_indices]
    y_test = y[test_indices]
    rf_oof_metrics = regression_metrics(y_train, oof_base[:, 0])
    gbr_oof_metrics = regression_metrics(y_train, oof_base[:, 1])

    print("\n" + "=" * 78)
    print("Generating stacking meta-model OOF predictions...")
    stacking_oof = generate_stacking_meta_oof(
        oof_base=oof_base,
        y_train=y_train,
        fold_assignments=fold_assignments,
    )
    stacking_oof_metrics = regression_metrics(y_train, stacking_oof)

    print("\n" + "=" * 78)
    print("Training the final RF, GBR, and Ridge models on the full training set...")
    final_objects = fit_final_stacking_model(
        feature_df=feature_df,
        y=y,
        train_indices=train_indices,
        test_indices=test_indices,
        oof_base=oof_base,
    )

    fullfit_train_prediction = final_objects["fullfit_train_prediction"]
    test_prediction = final_objects["test_prediction"]
    fullfit_train_metrics = regression_metrics(
        y_train,
        fullfit_train_prediction,
    )
    test_metrics = regression_metrics(y_test, test_prediction)

    train_results = build_prediction_results(
        filtered_indices=train_indices,
        original_indices=original_row_indices[train_indices],
        y_true=y_train,
        y_pred=fullfit_train_prediction,
        split_name="Training",
        prediction_type="Final_FullFit_Training",
    )
    test_results = build_prediction_results(
        filtered_indices=test_indices,
        original_indices=original_row_indices[test_indices],
        y_true=y_test,
        y_pred=test_prediction,
        split_name="Testing",
        prediction_type="Independent_Test",
    )

    train_results.to_csv(
        os.path.join(OUTPUT_DIR, OUTPUT_FILES["train_csv"]),
        index=False,
        encoding="utf-8-sig",
    )
    test_results.to_csv(
        os.path.join(OUTPUT_DIR, OUTPUT_FILES["test_csv"]),
        index=False,
        encoding="utf-8-sig",
    )

    ad_df = calculate_applicability_domain(
        x_train=final_objects["x_train"],
        x_test=final_objects["x_test"],
        train_indices=train_indices,
        test_indices=test_indices,
        original_row_indices=original_row_indices,
        y_train=y_train,
        train_prediction=fullfit_train_prediction,
        y_test=y_test,
        test_prediction=test_prediction,
    )
    ad_df.to_csv(
        os.path.join(OUTPUT_DIR, OUTPUT_FILES["ad_csv"]),
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
    print("Stacking training and evaluation completed")
    print(f"Random Forest OOF: {format_metrics(rf_oof_metrics)}")
    print(f"Gradient Boosting OOF: {format_metrics(gbr_oof_metrics)}")
    print(f"Stacking OOF: {format_metrics(stacking_oof_metrics)}")
    print(f"Final training set: {format_metrics(fullfit_train_metrics)}")
    print(f"Independent test set: {format_metrics(test_metrics)}")
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

    actual_files = sorted(
        filename
        for filename in os.listdir(OUTPUT_DIR)
        if os.path.isfile(os.path.join(OUTPUT_DIR, filename))
    )
    expected_files = sorted(OUTPUT_FILES.values())
    if actual_files != expected_files:
        raise RuntimeError(
            "The number or names of output files do not match the expected set.\n"
            f"Expected: {expected_files}\n"
            f"Actual: {actual_files}"
        )

    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("Generated output files:")
    for filename in actual_files:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
