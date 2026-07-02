import os
import random
import shutil
import warnings
from math import sqrt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.dataset import NumpySlicesDataset
from mindspore.nn import Adam
from mindspore.train import Model
from mindspore.train.callback import Callback
warnings.filterwarnings('ignore')
TARGET_COLUMN = 'logK'
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()
FEATURE_FILE = os.path.join(SCRIPT_DIR, 'NIST_features.csv')
TARGET_META_FILE = os.path.join(SCRIPT_DIR, 'NIST_targets.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'MLP_Final_Results')
TEST_SIZE = 0.2
RANDOM_STATE = 10
N_SPLITS = 5
CV_RANDOM_STATE = 10
MISSING_THRESHOLD = 0.6
CORRELATION_THRESHOLD = 0.9
MAX_CATEGORY_LEVELS = 40
ROBUST_CLIP_LIMIT = 10.0
HIDDEN_DIMS = [2, 94, 250]
DROPOUT_RATE = 0.192140940943413
LEARNING_RATE = 0.000486441491423032
WEIGHT_DECAY = 0.00473263434320598
BATCH_SIZE = 16
NUM_EPOCHS = 1000
MODEL_SEED = 42
AD_RESIDUAL_THRESHOLD = 3.0
OUTPUT_FILES = {'train_csv': 'MLP_Train_Results.csv', 'test_csv': 'MLP_Test_Results.csv', 'ad_csv': 'MLP_AD_Data.csv'}
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

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
    for encoding in ('utf-8-sig', 'gb18030', 'utf-8', 'latin1'):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'Unable to read file: {path}\nLast error: {last_error}')

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    return {'R2': float(r2_score(y_true, y_pred)), 'MAE': float(mean_absolute_error(y_true, y_pred)), 'MSE': float(mse), 'RMSE': float(sqrt(mse))}

def format_metrics(metrics):
    return f"R²={metrics['R2']:.6f}, MAE={metrics['MAE']:.6f}, MSE={metrics['MSE']:.6f}, RMSE={metrics['RMSE']:.6f}"

def remove_identity_and_leakage_columns(feature_df: pd.DataFrame) -> pd.DataFrame:
    drop_columns = {TARGET_COLUMN, 'logK', 'LogK', 'target', 'Target', 'y', 'SMILES', 'smiles', 'Canonical_SMILES', 'canonical_smiles', 'ligand_smiles', 'Ligand_SMILES', 'Ligand SMILES', 'ligand', 'Ligand', 'ligand_id', 'Ligand_ID', 'ligand_combination_key', 'Ligand_Combination_Key', 'ligand_combination', 'Ligand_Combination'}
    result = feature_df.drop(columns=[c for c in drop_columns if c in feature_df.columns], errors='ignore')
    index_like = [c for c in result.columns if str(c).lower().startswith('unnamed:') or str(c).lower() in {'index', 'row_index', 'sample_id', 'sampleid', 'id'}]
    return result.drop(columns=index_like, errors='ignore')

class TrainOnlyPreprocessor:

    def __init__(self, missing_threshold=0.6, correlation_threshold=0.9, max_category_levels=40, clip_limit=10.0):
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
        result = df.copy()
        for column in result.columns:
            if pd.api.types.is_numeric_dtype(result[column]):
                continue
            converted = pd.to_numeric(result[column], errors='coerce')
            n_non_missing = int(result[column].notna().sum())
            if n_non_missing == 0:
                continue
            if converted.notna().sum() / n_non_missing >= 0.95:
                result[column] = converted
        return result

    def fit(self, x_train_df: pd.DataFrame):
        self.numeric_columns = []
        self.categorical_columns = []
        self.encoder = None
        df = self._coerce_numeric_candidates(x_train_df)
        all_numeric = df.select_dtypes(include=[np.number, 'bool']).columns.tolist()
        numeric = df[all_numeric].copy()
        if numeric.shape[1] == 0:
            raise ValueError('No numeric features are available in the current training subset.')
        numeric = numeric.loc[:, numeric.isna().mean() <= self.missing_threshold]
        if numeric.shape[1] == 0:
            raise ValueError('No numeric features remain after missing-value filtering.')
        numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
        if numeric.shape[1] == 0:
            raise ValueError('No numeric features remain after constant-feature removal.')
        medians = numeric.median(numeric_only=True)
        numeric_imputed = numeric.fillna(medians)
        corr = numeric_imputed.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        correlated_to_drop = [column for column in upper.columns if (upper[column] > self.correlation_threshold).any()]
        numeric_imputed = numeric_imputed.drop(columns=correlated_to_drop, errors='ignore')
        if numeric_imputed.shape[1] == 0:
            raise ValueError('No numeric features remain after correlation filtering.')
        self.numeric_columns = numeric_imputed.columns.tolist()
        self.numeric_medians = medians[self.numeric_columns]
        self.scaler = RobustScaler()
        self.scaler.fit(numeric_imputed[self.numeric_columns])
        non_numeric = [c for c in df.columns if c not in all_numeric]
        self.categorical_columns = [c for c in non_numeric if 1 < int(df[c].nunique(dropna=True)) <= self.max_category_levels]
        if self.categorical_columns:
            try:
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            except TypeError:
                self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
            categorical_fit = df[self.categorical_columns].fillna('__MISSING__').astype(str)
            self.encoder.fit(categorical_fit)
        return self

    def transform(self, x_df: pd.DataFrame) -> np.ndarray:
        df = self._coerce_numeric_candidates(x_df)
        numeric = pd.DataFrame(index=df.index)
        for column in self.numeric_columns:
            if column in df.columns:
                numeric[column] = pd.to_numeric(df[column], errors='coerce')
            else:
                numeric[column] = np.nan
        numeric = numeric.fillna(self.numeric_medians)
        x_numeric = self.scaler.transform(numeric[self.numeric_columns])
        x_numeric = np.clip(x_numeric, -self.clip_limit, self.clip_limit)
        parts = [x_numeric]
        if self.categorical_columns:
            categorical = pd.DataFrame(index=df.index)
            for column in self.categorical_columns:
                if column in df.columns:
                    categorical[column] = df[column]
                else:
                    categorical[column] = '__MISSING__'
            categorical = categorical.fillna('__MISSING__').astype(str)
            parts.append(self.encoder.transform(categorical[self.categorical_columns]))
        x_out = np.hstack(parts).astype(np.float32)
        if not np.all(np.isfinite(x_out)):
            raise ValueError('NaN or infinite values remain after preprocessing.')
        return x_out

    def fit_transform(self, x_train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(x_train_df).transform(x_train_df)

class ResidualBlock(nn.Cell):

    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super().__init__()
        self.fc = nn.Dense(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(keep_prob=1.0 - dropout_rate)
        self.shortcut = nn.Dense(in_features, out_features) if in_features != out_features else None

    def construct(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        return out + shortcut

class ImprovedMLP(nn.Cell):

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super().__init__()
        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(ResidualBlock(previous_dim, hidden_dim, dropout_rate))
            previous_dim = hidden_dim
        layers.append(nn.Dense(previous_dim, output_dim))
        self.model = nn.SequentialCell(layers)

    def construct(self, x):
        return self.model(x)

class EpochLossRecorder(Callback):

    def __init__(self, model_ms, network, x_validation=None, y_validation=None, total_epochs=1000, print_every=20):
        super().__init__()
        self.model_ms = model_ms
        self.network = network
        self.total_epochs = total_epochs
        self.print_every = print_every
        self.train_losses = []
        self.validation_losses = []
        self._step_losses = []
        if x_validation is not None:
            self.x_validation = Tensor(np.asarray(x_validation, dtype=np.float32), ms.float32)
            self.y_validation = np.asarray(y_validation, dtype=np.float32).reshape(-1)
        else:
            self.x_validation = None
            self.y_validation = None

    @staticmethod
    def _to_float(value):
        try:
            return float(np.asarray(value.asnumpy()).reshape(-1)[0])
        except AttributeError:
            return float(np.asarray(value).reshape(-1)[0])

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        self._step_losses.append(self._to_float(cb_params.net_outputs))

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = int(cb_params.cur_epoch_num)
        if self._step_losses:
            train_loss = float(np.mean(self._step_losses))
        else:
            train_loss = self._to_float(cb_params.net_outputs)
        self._step_losses = []
        self.train_losses.append(train_loss)
        validation_loss = None
        if self.x_validation is not None:
            self.network.set_train(False)
            prediction = self.model_ms.predict(self.x_validation).asnumpy().reshape(-1)
            self.network.set_train(True)
            validation_loss = float(mean_squared_error(self.y_validation, prediction))
            self.validation_losses.append(validation_loss)
        if epoch == 1 or epoch % self.print_every == 0 or epoch == self.total_epochs:
            if validation_loss is None:
                print(f'Epoch {epoch:4d}/{self.total_epochs} | training MSE loss = {train_loss:.8f}')
            else:
                print(f'Epoch {epoch:4d}/{self.total_epochs} | training MSE loss = {train_loss:.8f} | validation MSE = {validation_loss:.8f}')

def train_one_model(x_train, y_train, x_predict, seed, model_tag, x_validation=None, y_validation=None):
    set_all_seeds(seed)
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train_2d = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    x_predict = np.asarray(x_predict, dtype=np.float32)
    network = ImprovedMLP(input_dim=x_train.shape[1], hidden_dims=HIDDEN_DIMS, output_dim=1, dropout_rate=DROPOUT_RATE)
    optimizer = Adam(network.trainable_params(), learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    drop_remainder = len(x_train) % BATCH_SIZE == 1
    if drop_remainder:
        print('The final batch contains one sample and was dropped because BatchNorm requires more than one sample.')
    train_dataset = NumpySlicesDataset((x_train, y_train_2d), shuffle=True).batch(BATCH_SIZE, drop_remainder=drop_remainder)
    model_ms = Model(network=network, loss_fn=nn.MSELoss(), optimizer=optimizer)
    recorder = EpochLossRecorder(model_ms=model_ms, network=network, x_validation=x_validation, y_validation=y_validation, total_epochs=NUM_EPOCHS, print_every=20)
    print(f'Training {model_tag} | seed={seed} | epochs={NUM_EPOCHS}')
    model_ms.train(NUM_EPOCHS, train_dataset, callbacks=[recorder], dataset_sink_mode=False)
    network.set_train(False)
    train_prediction = model_ms.predict(Tensor(x_train, ms.float32)).asnumpy().reshape(-1)
    predict_prediction = model_ms.predict(Tensor(x_predict, ms.float32)).asnumpy().reshape(-1)
    return {'train_prediction': train_prediction, 'predict_prediction': predict_prediction, 'train_losses': recorder.train_losses, 'validation_losses': recorder.validation_losses}

def applicability_domain_analysis(x_train, x_test, y_train, pred_train, y_test, pred_test, train_indices, test_indices, original_row_indices):
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    pred_train = np.asarray(pred_train, dtype=float).reshape(-1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)
    pred_test = np.asarray(pred_test, dtype=float).reshape(-1)
    train_residual = y_train - pred_train
    test_residual = y_test - pred_test
    residual_std = float(np.std(train_residual))
    if residual_std < 1e-12:
        residual_std = 1e-12
    train_standardized = train_residual / residual_std
    test_standardized = test_residual / residual_std
    train_average_distance = np.mean(cdist(x_train, x_train, metric='euclidean'), axis=1)
    test_average_distance = np.mean(cdist(x_test, x_train, metric='euclidean'), axis=1)
    train_outlier = np.abs(train_standardized) > AD_RESIDUAL_THRESHOLD
    test_outlier = np.abs(test_standardized) > AD_RESIDUAL_THRESHOLD
    train_df = pd.DataFrame({'Split': 'Training', 'Filtered_Row_Index': np.asarray(train_indices, dtype=int), 'Original_Row_Index': original_row_indices[train_indices], 'Experimental_logK': y_train, 'Predicted_logK': pred_train, 'Residual_True_minus_Predicted': train_residual, 'Average_Feature_Distance': train_average_distance, 'Standardized_Residual': train_standardized, 'AD_Residual_Threshold': AD_RESIDUAL_THRESHOLD, 'Outside_Applicability_Domain': train_outlier, 'Outside_Direction': np.where(train_standardized > AD_RESIDUAL_THRESHOLD, 'Above_+3', np.where(train_standardized < -AD_RESIDUAL_THRESHOLD, 'Below_-3', 'Inside')), 'Training_Residual_Standard_Deviation': residual_std})
    test_df = pd.DataFrame({'Split': 'Testing', 'Filtered_Row_Index': np.asarray(test_indices, dtype=int), 'Original_Row_Index': original_row_indices[test_indices], 'Experimental_logK': y_test, 'Predicted_logK': pred_test, 'Residual_True_minus_Predicted': test_residual, 'Average_Feature_Distance': test_average_distance, 'Standardized_Residual': test_standardized, 'AD_Residual_Threshold': AD_RESIDUAL_THRESHOLD, 'Outside_Applicability_Domain': test_outlier, 'Outside_Direction': np.where(test_standardized > AD_RESIDUAL_THRESHOLD, 'Above_+3', np.where(test_standardized < -AD_RESIDUAL_THRESHOLD, 'Below_-3', 'Inside')), 'Training_Residual_Standard_Deviation': residual_std})
    return pd.concat([train_df, test_df], ignore_index=True)

def build_prediction_results(indices, original_row_indices, y_true, y_pred, split_name, metrics):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    residual = y_true - y_pred
    return pd.DataFrame({'Split': split_name, 'Filtered_Row_Index': np.asarray(indices, dtype=int), 'Original_Row_Index': original_row_indices[indices], 'Experimental_logK': y_true, 'Predicted_logK': y_pred, 'Residual_True_minus_Predicted': residual, 'Absolute_Error': np.abs(residual), 'Squared_Error': residual ** 2, 'Dataset_R2': metrics['R2'], 'Dataset_MAE': metrics['MAE'], 'Dataset_MSE': metrics['MSE'], 'Dataset_RMSE': metrics['RMSE']})

def main():
    initialize_output_directory()
    print('Loading data...')
    feature_raw = read_csv_flexible(FEATURE_FILE)
    meta_raw = read_csv_flexible(TARGET_META_FILE)
    if TARGET_COLUMN not in meta_raw.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' was not found.")
    if len(feature_raw) != len(meta_raw):
        raise ValueError(f'The feature file contains {len(feature_raw)} rows, whereas the target file contains {len(meta_raw)} rows.')
    y = pd.to_numeric(meta_raw[TARGET_COLUMN], errors='coerce').to_numpy(dtype=float)
    original_row_indices = np.arange(len(y), dtype=int)
    valid_mask = np.isfinite(y)
    if not np.all(valid_mask):
        n_invalid = int(np.sum(~valid_mask))
        print(f'Removed samples with missing or nonnumeric targets: {n_invalid}')
        feature_raw = feature_raw.loc[valid_mask].reset_index(drop=True)
        y = y[valid_mask]
        original_row_indices = original_row_indices[valid_mask]
    feature_df = remove_identity_and_leakage_columns(feature_raw.copy())
    if feature_df.shape[1] == 0:
        raise ValueError('No candidate features remain after removing target, SMILES, and identifier columns.')
    all_indices = np.arange(len(y), dtype=int)
    train80_indices, test20_indices = train_test_split(all_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
    print(f'Valid samples: {len(y)}')
    print('Outer split: random 80% training set and 20% independent test set')
    print('Internal validation: KFold(n_splits=5) within the training set')
    print('Early stopping: disabled; each fold model and the final model are trained for 1000 epochs')
    print(f'Training samples: {len(train80_indices)}')
    print(f'Independent test samples: {len(test20_indices)}')
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    oof_prediction = np.full(len(train80_indices), np.nan, dtype=float)
    fold_metric_rows = []
    all_fold_train_losses = []
    all_fold_validation_losses = []
    for fold_number, (fold_train_local, fold_valid_local) in enumerate(kfold.split(train80_indices), start=1):
        fold_train_indices = train80_indices[fold_train_local]
        fold_valid_indices = train80_indices[fold_valid_local]
        print('\n' + '=' * 78)
        print(f'Fold {fold_number}/{N_SPLITS}')
        print(f'fold_train={len(fold_train_indices)}, fold_validation={len(fold_valid_indices)}')
        preprocessor = TrainOnlyPreprocessor(missing_threshold=MISSING_THRESHOLD, correlation_threshold=CORRELATION_THRESHOLD, max_category_levels=MAX_CATEGORY_LEVELS, clip_limit=ROBUST_CLIP_LIMIT)
        x_fold_train = preprocessor.fit_transform(feature_df.iloc[fold_train_indices])
        x_fold_valid = preprocessor.transform(feature_df.iloc[fold_valid_indices])
        print(f'Input features in this fold: {x_fold_train.shape[1]}')
        result = train_one_model(x_train=x_fold_train, y_train=y[fold_train_indices], x_predict=x_fold_valid, seed=MODEL_SEED + fold_number, model_tag=f'internal_fold_{fold_number}', x_validation=x_fold_valid, y_validation=y[fold_valid_indices])
        fold_train_prediction = result['train_prediction']
        fold_valid_prediction = result['predict_prediction']
        all_fold_train_losses.append(result['train_losses'])
        all_fold_validation_losses.append(result['validation_losses'])
        train_metrics = regression_metrics(y[fold_train_indices], fold_train_prediction)
        validation_metrics = regression_metrics(y[fold_valid_indices], fold_valid_prediction)
        print(f'Fold {fold_number} training subset: {format_metrics(train_metrics)}')
        print(f'Fold {fold_number} validation subset: {format_metrics(validation_metrics)}')
        oof_prediction[fold_valid_local] = fold_valid_prediction
        row = {'fold': fold_number, 'n_train': len(fold_train_indices), 'n_validation': len(fold_valid_indices)}
        for prefix, metrics in (('train', train_metrics), ('validation', validation_metrics)):
            for name, value in metrics.items():
                row[f'{prefix}_{name}'] = value
        fold_metric_rows.append(row)
    if np.any(~np.isfinite(oof_prediction)):
        raise RuntimeError('Some training samples did not receive out-of-fold predictions.')
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    oof_metrics = regression_metrics(y[train80_indices], oof_prediction)
    print('\n' + '=' * 78)
    print('Training the final model on the full 80% training set for 1000 epochs')
    print('The independent 20% test set is not used for preprocessing fitting or model training')
    final_preprocessor = TrainOnlyPreprocessor(missing_threshold=MISSING_THRESHOLD, correlation_threshold=CORRELATION_THRESHOLD, max_category_levels=MAX_CATEGORY_LEVELS, clip_limit=ROBUST_CLIP_LIMIT)
    x_train80 = final_preprocessor.fit_transform(feature_df.iloc[train80_indices])
    x_test20 = final_preprocessor.transform(feature_df.iloc[test20_indices])
    print(f'Final model input features: {x_train80.shape[1]}')
    final_result = train_one_model(x_train=x_train80, y_train=y[train80_indices], x_predict=x_test20, seed=MODEL_SEED, model_tag='final_train80', x_validation=None, y_validation=None)
    final_train_prediction = final_result['train_prediction']
    final_test_prediction = final_result['predict_prediction']
    final_train_metrics = regression_metrics(y[train80_indices], final_train_prediction)
    final_test_metrics = regression_metrics(y[test20_indices], final_test_prediction)
    train_results = build_prediction_results(indices=train80_indices, original_row_indices=original_row_indices, y_true=y[train80_indices], y_pred=final_train_prediction, split_name='Training', metrics=final_train_metrics)
    train_results.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILES['train_csv']), index=False, encoding='utf-8-sig')
    test_results = build_prediction_results(indices=test20_indices, original_row_indices=original_row_indices, y_true=y[test20_indices], y_pred=final_test_prediction, split_name='Testing', metrics=final_test_metrics)
    test_results.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILES['test_csv']), index=False, encoding='utf-8-sig')
    ad_data = applicability_domain_analysis(x_train=x_train80, x_test=x_test20, y_train=y[train80_indices], pred_train=final_train_prediction, y_test=y[test20_indices], pred_test=final_test_prediction, train_indices=train80_indices, test_indices=test20_indices, original_row_indices=original_row_indices)
    ad_data.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_FILES['ad_csv']), index=False, encoding='utf-8-sig')
    train_ad = ad_data[ad_data['Split'] == 'Training']
    test_ad = ad_data[ad_data['Split'] == 'Testing']
    train_outside = int(train_ad['Outside_Applicability_Domain'].sum())
    test_outside = int(test_ad['Outside_Applicability_Domain'].sum())
    print('\n' + '=' * 78)
    print('Random 80/20 split and five-fold cross-validation completed')
    print('Early stopping was not used; each fold model and the final model were trained for 1000 epochs')
    print('\nMean five-fold training metrics:')
    print(f"R²={fold_metrics_df['train_R2'].mean():.6f} ± {fold_metrics_df['train_R2'].std(ddof=1):.6f}, MAE={fold_metrics_df['train_MAE'].mean():.6f} ± {fold_metrics_df['train_MAE'].std(ddof=1):.6f}, MSE={fold_metrics_df['train_MSE'].mean():.6f} ± {fold_metrics_df['train_MSE'].std(ddof=1):.6f}, RMSE={fold_metrics_df['train_RMSE'].mean():.6f} ± {fold_metrics_df['train_RMSE'].std(ddof=1):.6f}")
    print('Mean five-fold validation metrics:')
    print(f"R²={fold_metrics_df['validation_R2'].mean():.6f} ± {fold_metrics_df['validation_R2'].std(ddof=1):.6f}, MAE={fold_metrics_df['validation_MAE'].mean():.6f} ± {fold_metrics_df['validation_MAE'].std(ddof=1):.6f}, MSE={fold_metrics_df['validation_MSE'].mean():.6f} ± {fold_metrics_df['validation_MSE'].std(ddof=1):.6f}, RMSE={fold_metrics_df['validation_RMSE'].mean():.6f} ± {fold_metrics_df['validation_RMSE'].std(ddof=1):.6f}")
    print('Overall OOF performance within the 80% training set:')
    print(format_metrics(oof_metrics))
    print('Final training set:')
    print(format_metrics(final_train_metrics))
    print('Independent 20% test set:')
    print(format_metrics(final_test_metrics))
    train_above = int((train_ad['Standardized_Residual'] > AD_RESIDUAL_THRESHOLD).sum())
    train_below = int((train_ad['Standardized_Residual'] < -AD_RESIDUAL_THRESHOLD).sum())
    test_above = int((test_ad['Standardized_Residual'] > AD_RESIDUAL_THRESHOLD).sum())
    test_below = int((test_ad['Standardized_Residual'] < -AD_RESIDUAL_THRESHOLD).sum())
    print(f'\nTraining samples outside the applicability domain: {train_outside}/{len(train_ad)} samples; >+3: {train_above}, <-3: {train_below}')
    print(f'Test samples outside the applicability domain: {test_outside}/{len(test_ad)} samples; >+3: {test_above}, <-3: {test_below}')
    actual_files = sorted((name for name in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, name))))
    expected_files = sorted(OUTPUT_FILES.values())
    if actual_files != expected_files:
        raise RuntimeError(f'The number or names of output files do not match the expected set.\nExpected: {expected_files}\nActual: {actual_files}')
    print(f'\nOutput directory: {os.path.abspath(OUTPUT_DIR)}')
    print('Generated 3 output files:')
    for filename in actual_files:
        print(f'  - {filename}')
if __name__ == '__main__':
    main()
