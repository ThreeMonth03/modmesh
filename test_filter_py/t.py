import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from filterpy.kalman import KalmanFilter

def solve_navigation():
    # --- 1. 定義常數 (來自 README) ---
    # DCM: IMU -> CON
    DCM_CON_IMU = np.array([
        [-0.2477, -0.1673,  0.9543],
        [-0.0478,  0.9859,  0.1604],
        [-0.9677, -0.0059, -0.2522]
    ])
    
    # ECEF 重力向量 (近似值，實際隨位置改變，此處簡化處理)
    # 注意：在 ECEF 中，重力方向指向地心
    G_ECEF = 9.80665 

    # --- 2. 讀取數據 ---
    df_dlc = pd.read_csv('./download/Flight1_Catered_Dataset-20201013/Data/dlc.csv')
    df_gt = pd.read_csv('./download/Flight1_Catered_Dataset-20201013/Data/truth.csv')

    # 時間對齊：以 DLC 為基準
    dlc_time = df_dlc.iloc[:, 0].values
    gt_time = df_gt.iloc[:, 0].values

    # 插值：將 GT 的四元數與位置同步到 DLC 的時間點
    # README 提到四元數是 [v0, v1, v2, s] -> 這是 Scipy 預設格式
    gt_quat_func = interp1d(gt_time, df_gt.iloc[:, 7:11].values, axis=0, fill_value="extrapolate")
    gt_pos_func = interp1d(gt_time, df_gt.iloc[:, 1:4].values, axis=0, fill_value="extrapolate")
    gt_vel_func = interp1d(gt_time, df_gt.iloc[:, 4:7].values, axis=0, fill_value="extrapolate")

    synced_quat = gt_quat_func(dlc_time)
    synced_pos_gt = gt_pos_func(dlc_time)
    synced_vel_gt = gt_vel_func(dlc_time)

    # --- 3. 初始化卡爾曼濾波器 ---
    # 狀態 x = [px, py, pz, vx, vy, vz]
    kf = KalmanFilter(dim_x=6, dim_z=6)
    kf.x = np.hstack([synced_pos_gt[0], synced_vel_gt[0]]) # 用第一筆 GT 初始化
    kf.P *= 0.1
    kf.R = np.eye(6) * 0.01 # 測量噪聲
    kf.Q = np.eye(6) * 0.001 # 過程噪聲

    results = []

    # --- 4. 核心迴圈 ---
    for i in range(1, len(dlc_time)):
        dt = (dlc_time[i] - dlc_time[i-1]) / 1e9
        if dt <= 0: continue

        # A. 取得 DLC 的 Delta Velocity (IMU 座標系)
        dv_imu = df_dlc.iloc[i, 1:4].values

        # B. 座標轉換：IMU -> CON
        dv_con = DCM_CON_IMU @ dv_imu

        # C. 座標轉換：CON -> ECEF (使用當前 GT 四元數)
        r_ecef_con = R.from_quat(synced_quat[i])
        dv_ecef = r_ecef_con.apply(dv_con)

        # D. 重力補償 (簡單模型：假設重力朝向 ECEF 中心反方向)
        # 嚴謹做法需根據當前位置計算重力向量，這裡示範補償邏輯
        pos_norm = kf.x[0:3] / np.linalg.norm(kf.x[0:3])
        gravity_step = -pos_norm * G_ECEF * dt
        
        # 預測預期速度增量 (IMU 量到的包含重力，所以要加回去補償)
        # 這裡我們直接把 dv_ecef 當作控制輸入
        accel_ecef = (dv_ecef / dt) + (pos_norm * G_ECEF) 

        # E. 卡爾曼預測 (F 矩陣)
        kf.F = np.eye(6)
        kf.F[0:3, 3:6] = np.eye(3) * dt
        
        # 使用 IMU 數據作為 Prediction
        # x = Fx + Bu, 其中 u 是加速度
        B = np.zeros((6, 3))
        B[0:3, 0:3] = 0.5 * dt**2 * np.eye(3)
        B[3:6, 0:3] = dt * np.eye(3)
        
        kf.predict(u=accel_ecef, B=B)

        # F. 卡爾曼更新 (使用 GT 進行校正)
        # 在實際應用中，這裡會放 Lidar 或 GPS，現在用 GT 來模擬「觀測」
        z = np.hstack([synced_pos_gt[i], synced_vel_gt[i]])
        kf.update(z)

        results.append(kf.x.copy())

    return np.array(results), synced_pos_gt

# 執行
estimated, actual = solve_navigation()
print("估計完成，最後誤差(m):", np.linalg.norm(estimated[-1, :3] - actual[-1, :3]))