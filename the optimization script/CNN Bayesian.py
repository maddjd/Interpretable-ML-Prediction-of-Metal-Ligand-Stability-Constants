import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from feature_selector import FeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
import mindspore as ms
from mindspore import context, Tensor, nn
from mindspore.dataset import NumpySlicesDataset
from mindspore.nn import MSELoss, Adam
from mindspore.train.callback import Callback, LossMonitor, EarlyStopping
from mindspore.train import Model
from scipy.spatial.distance import cdist
import optuna  #  Optuna


# MindSpore
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

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

# 
fs = FeatureSelector(data=x, labels=y)
fs.identify_missing(missing_threshold=0.6)
fs.identify_single_unique()
fs.identify_collinear(correlation_threshold=0.95)

# 
train_no_missing = fs.remove(methods=['missing', 'single_unique', 'collinear'], keep_one_hot=False)
x1 = pd.DataFrame(train_no_missing)

# 
scaler = RobustScaler()
x1 = scaler.fit_transform(x1)
x1 = np.clip(x1, -10, 10)
# x1_df = pd.DataFrame(x1, columns=train_no_missing.columns)
# x1_df.to_csv("standardized_feature_data.csv", index=False)
# print(" 'standardized_feature_data.csv'")
y = y.astype(np.float32)

# 
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=10)

# CNN
x_train = np.expand_dims(x_train, axis=1).astype(np.float32)  #  (batch_size, channels, width)
x_test = np.expand_dims(x_test, axis=1).astype(np.float32)
y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
y_test = np.array(y_test, dtype=np.float32).reshape(-1, 1)

# CNN
class CNN(nn.Cell):
    def __init__(self, input_width, out_channels, kernel_size, fc1_units, dropout_rate):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, out_channels, kernel_size=kernel_size, pad_mode='valid', has_bias=True)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        #  Dense 
        self.fc1_input_size = out_channels * (input_width - kernel_size + 1)
        self.fc1 = nn.Dense(self.fc1_input_size, fc1_units)
        self.fc2 = nn.Dense(fc1_units, 1)
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 
def objective(trial):
    # 
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.3)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    out_channels = trial.suggest_int('out_channels', 16, 128)  # 
    kernel_size = trial.suggest_int('kernel_size', 2, 5)  # 
    fc1_units = trial.suggest_int('fc1_units', 16, 128)  # 

    # 
    input_width = x_train.shape[2]
    if input_width - kernel_size + 1 <= 0:
        return float('-inf')  #  <= 0，

    # 
    model = CNN(input_width=input_width, out_channels=out_channels, kernel_size=kernel_size,
                fc1_units=fc1_units, dropout_rate=dropout_rate)

    # 
    optimizer = Adam(model.trainable_params(), learning_rate=learning_rate, weight_decay=weight_decay)
    loss_fn = MSELoss()

    # 
    train_dataset = NumpySlicesDataset((x_train, y_train), shuffle=True).batch(batch_size)
    test_dataset = NumpySlicesDataset((x_test, y_test), shuffle=False).batch(batch_size)

    #  Model
    model_ms = Model(network=model, loss_fn=loss_fn, optimizer=optimizer, metrics={'MSE': nn.MSE()})

    # 
    early_stopping = EarlyStopping(monitor="loss", patience=10, min_delta=0.0001, verbose=False)
    loss_monitor = LossMonitor()

    # 
    num_epochs = 1000
    try:
        model_ms.train(num_epochs, train_dataset, callbacks=[loss_monitor, early_stopping], dataset_sink_mode=False)
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('-inf')  # ，

    # （ R²）
    y_test_pred = model_ms.predict(Tensor(x_test, ms.float32)).asnumpy().squeeze()
    r2 = r2_score(y_test, y_test_pred)

    return r2

# 
study = optuna.create_study(direction='maximize')  # ： R²
study.optimize(objective, n_trials=50)  #  50 

# 
print("：")
best_trial = study.best_trial
print(f"   R²: {best_trial.value}")
print("  ：")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

