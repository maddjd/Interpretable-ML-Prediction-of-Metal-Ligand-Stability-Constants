import os
import numpy as np
import pandas as pd
from feature_selector import FeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.dataset import NumpySlicesDataset
from mindspore.train.callback import EarlyStopping, LossMonitor, Callback
from mindspore.train import Model
from mindspore.nn import MSELoss, HuberLoss
from mindspore.nn import Adam
import optuna
from math import sqrt
from scipy.spatial.distance import cdist
from scipy.stats import mstats
import warnings
warnings.filterwarnings('ignore')



context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Data loading

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

FEATURE_FILE = os.path.join(SCRIPT_DIR, "Dataset_features.csv")
TARGET_META_FILE = os.path.join(SCRIPT_DIR, "Dataset_targets.csv")
TARGET_COLUMN = "logK"

x = pd.read_csv(FEATURE_FILE)
mdf = pd.read_csv(TARGET_META_FILE)
y = mdf[TARGET_COLUMN]


fs = FeatureSelector(data=x, labels=y)
fs.identify_missing(missing_threshold=0.6)
fs.identify_single_unique()
fs.identify_collinear(correlation_threshold=0.95)

train_no_missing = fs.remove(methods=['missing', 'single_unique', 'collinear'], keep_one_hot=False)
x1 = pd.DataFrame(train_no_missing)



scaler = RobustScaler() # StandardScaler
x1 = scaler.fit_transform(x1)
x1 = np.clip(x1, -10, 10)
# x1_df = pd.DataFrame(x1, columns=train_no_missing.columns)
y = y.astype(np.float32)


x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=0)

x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32).reshape(-1, 1)

print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

train_dataset = NumpySlicesDataset((x_train, y_train), shuffle=True)
test_dataset = NumpySlicesDataset((x_test, y_test), shuffle=False)


x_train = np.nan_to_num(x_train, nan=0.0, posinf=1e5, neginf=-1e5)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=1e5, neginf=-1e5)
x_test = np.nan_to_num(x_test, nan=0.0, posinf=1e5, neginf=-1e5)
y_test = np.nan_to_num(y_test, nan=0.0, posinf=1e5, neginf=-1e5)

batch_size = 32
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# class MLP(nn.Cell):
#     def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.0):
#         super(MLP, self).__init__()
#         layers = []
#         for i, hidden_dim in enumerate(hidden_dims):
#             layers.append(nn.Dense(input_dim if i == 0 else hidden_dims[i-1], hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(p=dropout_rate))
#         layers.append(nn.Dense(hidden_dims[-1], output_dim))
#         self.model = nn.SequentialCell(layers)
#
#     def construct(self, x):
#         return self.model(x)

class ResidualBlock(nn.Cell):
    def __init__(self, in_features, out_features, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Dense(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)
        if in_features != out_features:
            self.shortcut = nn.Dense(in_features, out_features)
        else:
            self.shortcut = None

    def construct(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        return out + shortcut

class ImprovedMLP(nn.Cell):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super(ImprovedMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(ResidualBlock(prev_dim, hd, dropout_rate))
            prev_dim = hd
        layers.append(nn.Dense(prev_dim, output_dim))
        self.model = nn.SequentialCell(layers)

    def construct(self, x):
        return self.model(x)


def objective(trial):
    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-3,
        log=True
    )

    dropout_rate = trial.suggest_float(
        "dropout_rate",
        0.0,
        0.3
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-2,
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [8, 16, 32, 64]
    )

    n_layers = trial.suggest_int(
        "n_layers",
        1,
        3
    )

    hidden_dims = []

    for i in range(n_layers):
        hidden_dim = trial.suggest_int(
            f"hidden_dim_layer_{i}",
            16,
            128
        )
        hidden_dims.append(hidden_dim)

    input_dim = x_train.shape[1]
    output_dim = 1
    model = ImprovedMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout_rate=dropout_rate)

    optimizer = Adam(model.trainable_params(), learning_rate=learning_rate, weight_decay=weight_decay)
    loss_fn = MSELoss()

    train_dataset = NumpySlicesDataset((x_train, y_train), shuffle=True).batch(batch_size)
    test_dataset = NumpySlicesDataset((x_test, y_test), shuffle=False).batch(batch_size)

    model_ms = Model(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics={'MSE': nn.MSE()})

    early_stopping = EarlyStopping(monitor="loss", patience=40, min_delta=0.0001, verbose=False)
    loss_monitor = LossMonitor()

    num_epochs = 1000
    model_ms.train(num_epochs, train_dataset, callbacks=[loss_monitor, early_stopping], dataset_sink_mode=False)

    y_test_pred = model_ms.predict(Tensor(x_test, ms.float32)).asnumpy().squeeze()
    r2 = r2_score(y_test, y_test_pred)

    return r2


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)




best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# input_dim = x_train.shape[1]
# output_dim = 1
# dropout_rate = 0.2 #0.096
#
# model = ImprovedMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout_rate=dropout_rate)
#
#
# lr_schedule = nn.cosine_decay_lr(
# )
#
# loss_fn = HuberLoss()
# optimizer = Adam(model.trainable_params(), learning_rate=lr_schedule, weight_decay=0.001)
#
# def plot_loss_curve(train_losses, test_losses):
#     plt.figure(figsize=(10, 8), dpi=1200)
#     plt.plot(train_losses, label="Training Loss", linewidth=2)
#     plt.plot(test_losses, label="Testing Loss", linewidth=2, linestyle="--")
#     plt.xlabel("Epochs", fontsize=18)
#     plt.ylabel("Loss", fontsize=18)
#     plt.title("Training and Testing Loss Curve", fontsize=20)
#     plt.legend(fontsize=16)
#     plt.grid()
#     plt.tight_layout()
#     plt.show()
#
# def plot_scatter(true_train, pred_train, true_test, pred_test):
#     plt.figure(figsize=(8, 6), dpi=1200)
#     plt.scatter(true_train, pred_train, label="Train Data", alpha=0.6, color="blue")
#     plt.scatter(true_test, pred_test, label="Test Data", alpha=0.6, color="orange")
#     plt.plot(
#         [min(true_train), max(true_train)],
#         [min(true_train), max(true_train)],
#         color="red",
#         linestyle="--",
#         label="Ideal Fit",
#     )
#     plt.xlabel("True Values")
#     plt.ylabel("Predicted Values")
#     plt.title("Scatter Plot of True vs Predicted Values")
#     plt.legend()
#     plt.grid()
#     plt.show()
#
#
# def plot_error_distribution(y_true, y_pred):
#     errors = y_true - y_pred
#
#     plt.figure(figsize=(10, 6), dpi=1200)
#
#     plt.hist(errors, bins=30, alpha=0.6, edgecolor='k', density=True, label="Error Distribution")
#
#     sns.kdeplot(errors, color='r', label="Kernel Density Estimate", linewidth=2)
#
#     plt.axvline(x=0, color='k', linestyle='--', label="Zero Error")
#
#     plt.xlabel("Prediction Error (True - Predicted)", fontsize=14, fontname="Times New Roman")
#     plt.ylabel("Density", fontsize=14, fontname="Times New Roman")
#     plt.title("Error Distribution of Test Predictions", fontsize=16, fontname="Times New Roman")
#     plt.legend(fontsize=12)
#     plt.grid()
#
#     plt.tight_layout()
#     plt.show()
#
# class LossRecorderWithValidation(Callback):
#     def __init__(self):
#         self.train_losses = []
#         self.test_losses = []
#
#     def epoch_end(self, run_context):
#         cb_params = run_context.original_args()
#
#         y_test_pred = model_ms.predict(Tensor(x_test, ms.float32)).asnumpy().squeeze()
#         test_loss = mean_squared_error(y_test, y_test_pred)
#
# model_ms = Model(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics={'MSE': nn.MSE()})
#
# early_stopping = EarlyStopping(monitor="loss", patience=20, min_delta=0.0001, verbose=True)
#
# num_epochs = 1000
# loss_recorder = LossRecorderWithValidation()
#
# model_ms.train(num_epochs, train_dataset, callbacks=callbacks, dataset_sink_mode=False)
#
# # r2_scores = []
# # cv_results = []
# #
# # for fold, (train_index, val_index) in enumerate(kf.split(x_train), 1):
# #     x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
# #     y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
# #
# #     scaler = StandardScaler()
# #     x_train_fold = scaler.fit_transform(x_train_fold)
# #     x_val_fold = scaler.transform(x_val_fold)
# #
# #     model_fold = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout_rate=dropout_rate)
# #     model_ms_fold = Model(network=model_fold, loss_fn=loss_fn, optimizer=optimizer, metrics={'MSE': nn.MSE()})
# #
# #     model_ms_fold.train(num_epochs, NumpySlicesDataset((x_train_fold, y_train_fold)).batch(batch_size),
# #                         dataset_sink_mode=False)
# #
# #     y_val_pred = model_ms_fold.predict(Tensor(x_val_fold, ms.float32)).asnumpy().squeeze()
# #     r2 = r2_score(y_val_fold, y_val_pred)
# #     r2_scores.append(r2)
# #
# #     cv_results.append({"Fold": fold, "R2": r2})
# #     print(f"Fold {fold}, R2: {r2}")
# #
# # cv_results_df = pd.DataFrame(cv_results)
# #
# 
# # plot_loss_curve(loss_recorder.train_losses, loss_recorder.test_losses)
#
# y_train_pred = model_ms.predict(Tensor(x_train, ms.float32)).asnumpy().squeeze()
# y_test_pred = model_ms.predict(Tensor(x_test, ms.float32)).asnumpy().squeeze()
#
# def applicability_domain_combined_analysis(x_train, x_test, y_train, y_train_pred, y_test, y_test_pred):
#     """
#     """
#
#
#
#
#     plt.figure(figsize=(10, 8), dpi=1200)
#     plt.scatter(train_avg_distances, train_std_residuals, alpha=0.7, color='blue', label='Training Samples', s=50)
#     plt.scatter(test_avg_distances, test_std_residuals, alpha=0.7, color='orange', label='Testing Samples', s=50)
#     plt.axhline(y=3, color='red', linestyle='--', label='Residual Threshold (+3)')
#     plt.axhline(y=-3, color='red', linestyle='--', label='Residual Threshold (-3)')
#     plt.xlabel("Average Feature Distance", fontsize=16)
#     plt.ylabel("Standardized Residuals", fontsize=16)
#     plt.title("Applicability Domain Analysis (Training & Testing)", fontsize=18)
#     plt.legend(fontsize=14)
#     plt.grid()
#     plt.tight_layout()
#     plt.show()
#
#     train_outliers = np.where((train_std_residuals > 3) | (train_std_residuals < -3))
#     test_outliers = np.where((test_std_residuals > 3) | (test_std_residuals < -3))
#
#
#
# # applicability_domain_combined_analysis(
# #     x_train, x_test,
# #     y_train.squeeze(), y_train_pred,
# #     y_test.squeeze(), y_test_pred
# # )
#
# '---------------------------------'
# # plot_error_distribution(y_test.squeeze(), y_test_pred)
# #
# # plot_scatter(y_train, y_train_pred, y_test, y_test_pred)
# #
# # train_results = pd.DataFrame({'True': y_train.squeeze(), 'Predicted': y_train_pred})
# # test_results = pd.DataFrame({'True': y_test.squeeze(), 'Predicted': y_test_pred})
# #
#
# print("R2:", r2_score(y_train, y_train_pred))
# print("MAE:", mean_absolute_error(y_train, y_train_pred))
# print("MSE:", mean_squared_error(y_train, y_train_pred))
#
# print("R2:", r2_score(y_test, y_test_pred))
# print("MAE:", mean_absolute_error(y_test, y_test_pred))
# print("MSE:", mean_squared_error(y_test, y_test_pred))
