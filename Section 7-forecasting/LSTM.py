import keras   #cpu tensorflow
print(keras.__version__)
import tensorflow as tf
print(tf.__version__)

# -*- coding: utf-8 -*-
# 中国碳排放权交易日度面板 - LSTM模型（样本内一步预测）
# ============================================================
# 步骤：
# 1. 导入并筛选数据
# 2. 全样本训练
# 3. 超参数搜索
# 4. 一步样本内预测及误差分析
# 5. 三步样本外预测与置信区间
# 6. 扰动数据鲁棒性分析
# 7. 汇总结果

import os, time, math, itertools, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")

# ============== 0. 固定随机种子以复现 ==============
seed = 123
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# ============== 1. 数据导入与筛选 ==============
xlsx_path = "D:\TEST\computation economics\碳排放权交易日度面板.xlsx"
data = pd.read_excel(xlsx_path, engine="openpyxl")

# 删除第一行，与R的 data[-1, ] 对齐
data_1 = data.iloc[1:, :].copy()

# 选取指定列
cols = ["交易日期", "城市名称", "交易品种", "收盘价"]
data_2 = data_1[cols].copy()

# 筛选 广东 + GDEA
GD = data_2[(data_2["城市名称"] == "广东") & (data_2["交易品种"] == "GDEA")].copy()

# 去掉前7行，与 R 的 data1 = GD[8:nrow(GD),] 一致
data1 = GD.iloc[7:, :].copy()
data1["收盘价"] = pd.to_numeric(data1["收盘价"], errors="coerce")
data1 = data1.dropna(subset=["收盘价"]).reset_index(drop=True)

# 提取目标序列
S = np.log(data1["收盘价"].astype(float).values)
T = len(S)


# ============== 2. 数据标准化（Z-score，全样本） ==============
mean_S = np.mean(S)
sd_S = np.std(S, ddof=0) if np.std(S, ddof=0) > 0 else 1.0
S_scaled = (S - mean_S) / sd_S

# 构造输入输出：X_t = S_{t-1}, y_t = S_t
lag = 1
X = np.array([S_scaled[i-lag:i] for i in range(lag, len(S_scaled))])
y = S_scaled[lag:]

X_train = X.reshape(X.shape[0], lag, 1)   # (T-lag, lag, 1)
y_train = y.reshape(-1, 1)

print("训练样本数:", X_train.shape[0])

# ============== 3. 构建模型 + 超参数搜索 ==============
def build_model(units, lr, dropout):
    model = keras.Sequential([
        layers.Input(shape=(lag, 1)),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model

param_grid = {
    "units": [32, 64, 96],
    "lr": [0.001, 0.0005],
    "dropout": [0.1, 0.2, 0.3]
}
grid_list = list(itertools.product(param_grid["units"], param_grid["lr"], param_grid["dropout"]))

best_loss = np.inf
best_params = None

print("开始网格搜索超参数...")
for (units, lr, dropout) in grid_list:
    print(f"尝试参数: units={units}, lr={lr}, dropout={dropout}")
    model = build_model(units, lr, dropout)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
    hist = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=0, callbacks=[es])
    cur_best = np.min(hist.history["val_loss"])
    if cur_best < best_loss:
        best_loss = cur_best
        best_params = {"units": units, "lr": lr, "dropout": dropout}

print("\n---- 最优参数 ----")
print(best_params)

# ============== 4. 用最优参数重训模型 ==============
t1 = time.time()
model_best = build_model(best_params["units"], best_params["lr"], best_params["dropout"])
es2 = keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True, verbose=0)
model_best.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0, callbacks=[es2])
t2 = time.time()
train_time = t2 - t1
print(f"训练耗时: {train_time:.4f} 秒")

# ============== 5. 一步样本内预测与误差 ==============
t3 = time.time()
y_pred_scaled = model_best.predict(X_train, verbose=0).reshape(-1)
t4 = time.time()
pred_time = t4 - t3

y_true = y_train.reshape(-1) * sd_S + mean_S
y_pred = y_pred_scaled * sd_S + mean_S

# 估计 log 空间预测误差方差（同方差近似；可换滚动/异方差）
resid_log = y_true - y_pred
vhat = np.var(resid_log, ddof=1)

# 价格空间的点预测
S_true = np.exp(y_true)
S_hat_mean   = np.exp(y_pred + 0.5 * vhat)    # 无偏均值（推荐）

eps = 1e-12
mse = mean_squared_error(S_true, S_hat_mean)
rmse = math.sqrt(mse)
mae = mean_absolute_error(S_true, S_hat_mean)
rae = np.sum(np.abs(S_true - S_hat_mean)) / (np.sum(np.abs(S_true - np.mean(S_true))) + eps)
mdape = np.median(np.abs((S_true - S_hat_mean) / (np.abs(S_true) + eps)))

print("\n---- 一步样本内预测误差 ----")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"RAE  : {rae:.6f}")
print(f"MdAPE: {mdape:.6f}")
print(f"预测耗时: {pred_time:.6f} 秒")

#输出结果
# 假设 df_result 已经包含 S_true 和 S_hat_mean
# 如果还有日期信息，一并导出
df_result = pd.DataFrame({"S_true": S_true, "S_hat_mean": S_hat_mean})
output_path = r"D:\TEST\computation economics\LSTM_pred.csv"  # 用 r 前缀防止路径转义
df_result.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"✅ 已成功导出到: {output_path}")

#================ 区间估计诊断结果 =========================
# ============== 6. 区间估计诊断结果 =========================
import numpy as np

def interval_metrics(y_true, lower, upper):
    """区间估计诊断指标：PICP, PINAW, AWD"""
    y_true, lower, upper = np.array(y_true), np.array(lower), np.array(upper)
    N = len(y_true)
    ymax, ymin = np.nanmax(y_true), np.nanmin(y_true)

    # PICP
    picp = np.mean((y_true >= lower) & (y_true <= upper))
    # PINAW
    pinaw = np.mean(upper - lower) / (ymax - ymin + 1e-12)
    # AWD
    delta = np.where(
        y_true < lower, (lower - y_true) / (upper - lower + 1e-12),
        np.where(y_true > upper, (y_true - upper) / (upper - lower + 1e-12), 0)
    )
    awd = np.mean(delta)
    return dict(PICP=picp, PINAW=pinaw, AWD=awd)


# ---------- 残差标准差（log空间） ----------
sigma_hat = np.sqrt(vhat)

# ---------- 正态分位数 ----------
z_95 = 1.96
z_85 = 1.44
z_75 = 1.15

# ---------- 计算区间（在log空间取分位数，再指数化） ----------
L95 = np.exp(y_pred - z_95 * sigma_hat)
U95 = np.exp(y_pred + z_95 * sigma_hat)
L85 = np.exp(y_pred - z_85 * sigma_hat)
U85 = np.exp(y_pred + z_85 * sigma_hat)
L75 = np.exp(y_pred - z_75 * sigma_hat)
U75 = np.exp(y_pred + z_75 * sigma_hat)

# ---------- 区间诊断结果 ----------
metrics_95 = interval_metrics(S_true, L95, U95)
metrics_85 = interval_metrics(S_true, L85, U85)
metrics_75 = interval_metrics(S_true, L75, U75)

print("\n---- 区间估计诊断结果 ----")
print(f"α = 0.95 → PICP = {metrics_95['PICP']:.4f}, PINAW = {metrics_95['PINAW']:.4f}, AWD = {metrics_95['AWD']:.4f}")
print(f"α = 0.85 → PICP = {metrics_85['PICP']:.4f}, PINAW = {metrics_85['PINAW']:.4f}, AWD = {metrics_85['AWD']:.4f}")
print(f"α = 0.75 → PICP = {metrics_75['PICP']:.4f}, PINAW = {metrics_75['PINAW']:.4f}, AWD = {metrics_75['AWD']:.4f}")

# ======================================================
# 7️⃣ LSTM 三步样本外递推预测 + 区间估计 + 误差计算（重新划分训练集）
# ======================================================

steps = 3
lag = 1

# 划分样本：前 T-3 训练，后三步预测
S_train = S[:-steps]
S_test  = S[-steps:]

# --- 重新标准化训练集 ---
mean_S = np.mean(S_train)
sd_S = np.std(S_train, ddof=0) if np.std(S_train, ddof=0) > 0 else 1.0
S_train_scaled = (S_train - mean_S) / sd_S

# --- 构造训练样本 ---
X_train = np.array([S_train_scaled[i-lag:i] for i in range(lag, len(S_train_scaled))])
y_train = S_train_scaled[lag:]
X_train = X_train.reshape(X_train.shape[0], lag, 1)
y_train = y_train.reshape(-1, 1)

print(f"\n训练样本数: {X_train.shape[0]}, 测试样本长度: {len(S_test)}")

# --- 重新训练模型（可直接复用最优超参数）---
model_out = build_model(best_params["units"], best_params["lr"], best_params["dropout"])
es_out = keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True, verbose=0)
model_out.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0, callbacks=[es_out])

# ======================================================
# 递推三步预测（样本外）
# ======================================================
last_input = X_train[-1].reshape(1, lag, 1)
log_preds = []

for h in range(steps):
    next_scaled = model_out.predict(last_input, verbose=0)[0, 0]
    next_log = next_scaled * sd_S + mean_S
    log_preds.append(next_log)
    new_input_scaled = np.append(last_input[0, 1:, 0], next_scaled) if lag > 1 else np.array([next_scaled])
    last_input = new_input_scaled.reshape(1, lag, 1)

y_true = y_train.reshape(-1) * sd_S + mean_S
y_pred_scaled = model_best.predict(X_train, verbose=0).reshape(-1)
y_pred = y_pred_scaled * sd_S + mean_S
resid_log = y_true - y_pred
sigma_hat = np.std(resid_log, ddof=1)
# ======================================================
# 区间估计 + 无偏修正
# ======================================================
z_95, z_85, z_75 = 1.96, 1.44, 1.15
pred_price = np.exp(log_preds + 0.5 * sigma_hat**2)
true_price = np.exp(S_test)

L95 = np.exp(log_preds - z_95 * sigma_hat)
U95 = np.exp(log_preds + z_95 * sigma_hat)
L85 = np.exp(log_preds - z_85 * sigma_hat)
U85 = np.exp(log_preds + z_85 * sigma_hat)
L75 = np.exp(log_preds - z_75 * sigma_hat)
U75 = np.exp(log_preds + z_75 * sigma_hat)

# ======================================================
# 误差指标
# ======================================================
eps = 1e-12
MSE = mean_squared_error(true_price, pred_price)
RMSE = math.sqrt(MSE)
MAE = mean_absolute_error(true_price, pred_price)
RAE = np.sum(np.abs(true_price - pred_price)) / (np.sum(np.abs(true_price - np.mean(np.exp(S_train)))) + eps)
MdAPE = np.median(np.abs((true_price - pred_price) / (true_price + eps)))

print("\n===== LSTM 三步样本外预测误差 =====")
print(f"MSE  : {MSE:.6f}")
print(f"RMSE : {RMSE:.6f}")
print(f"MAE  : {MAE:.6f}")
print(f"RAE  : {RAE:.6f}")
print(f"MdAPE: {MdAPE:.6f}")


# ============== 6. 三步样本外预测 + 区间估计 ==============
steps_out = 3
ctx = np.array([S_scaled[-1]])  # 用最后一个标准化值作为起点
pred_out_scaled = []
for h in range(steps_out):
    x = ctx.reshape(1, 1, 1)
    yhat = model_best.predict(x, verbose=0)[0, 0]
    pred_out_scaled.append(yhat)
    ctx = np.array([yhat])

pred_out = np.array(pred_out_scaled) * sd_S + mean_S

# 95%区间（正态近似）
resid = y_true - y_pred
sigma = np.std(resid, ddof=1)
z = 1.96
lower = pred_out - z * sigma
upper = pred_out + z * sigma

print("\n(Ex-ante) 三步未来预测（含95%区间）：")
df_fore = pd.DataFrame({
    "Step": np.arange(1, steps_out + 1),
    "Forecast": np.round(pred_out, 6),
    "Lower95": np.round(lower, 6),
    "Upper95": np.round(upper, 6)
})
print(df_fore.to_string(index=False))

plt.figure(figsize=(6,4))
steps = np.arange(1, steps_out + 1)
plt.plot(steps, pred_out, marker="o", label="点预测")
plt.fill_between(steps, lower, upper, alpha=0.2, label="95%区间")
plt.xticks(steps, [f"步{i}" for i in steps])
plt.title("三步样本外预测（95%区间）")
plt.xlabel("预测步数")
plt.ylabel("收盘价")
plt.legend()
plt.tight_layout()
plt.show()

S_true = np.exp(y_true)
S_hat_mean   = np.exp(y_pred + 0.5 * vhat)    # 无偏均值（推荐）

eps = 1e-12
mse = mean_squared_error(S_true, S_hat_mean)
rmse = math.sqrt(mse)
mae = mean_absolute_error(S_true, S_hat_mean)
rae = np.sum(np.abs(S_true - S_hat_mean)) / (np.sum(np.abs(S_true - np.mean(S_true))) + eps)
mdape = np.median(np.abs((S_true - S_hat_mean) / (np.abs(S_true) + eps)))

print("\n---- 一步样本内预测误差 ----")
print(f"MSE  : {mse:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"MAE  : {mae:.6f}")
print(f"RAE  : {rae:.6f}")
print(f"MdAPE: {mdape:.6f}")
print(f"预测耗时: {pred_time:.6f} 秒")


# ============== 7. 扰动数据鲁棒性分析 ==============
noise_scale = 0.05 * np.std(y_train)
y_train_noisy = y_train + np.random.normal(0, noise_scale, size=y_train.shape).astype(np.float32)

model_noise = build_model(best_params["units"], best_params["lr"], best_params["dropout"])
model_noise.fit(X_train, y_train_noisy, epochs=100, batch_size=16, verbose=0)
y_pred_noisy_scaled = model_noise.predict(X_train, verbose=0).reshape(-1)
y_pred_noisy = y_pred_noisy_scaled * sd_S + mean_S

mse_p = mean_squared_error(y_true, y_pred_noisy)
rmse_p = math.sqrt(mse_p)
mae_p = mean_absolute_error(y_true, y_pred_noisy)
rae_p = np.sum(np.abs(y_true - y_pred_noisy)) / (np.sum(np.abs(y_true - np.mean(y_true))) + eps)
mdape_p = np.median(np.abs((y_true - y_pred_noisy) / (np.abs(y_true) + eps)))

print("\n---- 扰动数据后的误差 ----")
print(f"MSE  : {mse_p:.6f} (原 {mse:.6f})")
print(f"RMSE : {rmse_p:.6f} (原 {rmse:.6f})")
print(f"MAE  : {mae_p:.6f} (原 {mae:.6f})")
print(f"RAE  : {rae_p:.6f} (原 {rae:.6f})")
print(f"MdAPE: {mdape_p:.6f} (原 {mdape:.6f})")

# ============== 8. 汇总输出 ==============
print("\n===================== 总结 =====================")
print(f"最优参数: units={best_params['units']}, lr={best_params['lr']:.4f}, dropout={best_params['dropout']:.2f}")
print(f"训练耗时: {train_time:.4f} 秒")
print(f"预测耗时: {pred_time:.6f} 秒")
print(f"MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, RAE={rae:.6f}, MdAPE={mdape:.6f}")
print("===============================================")
