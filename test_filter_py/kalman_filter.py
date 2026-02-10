import numpy as np
from filterpy.kalman import KalmanFilter

from align_data import (
    align_and_transform,
    imu_pos_to_con_pos_ecef,
    imu_vec_to_ecef_variant,
)

class load_data():
    @staticmethod
    def load_dlc():
        # 讀取 CSV
        raw_data = np.genfromtxt('./download/Flight1_Catered_Dataset-20201013/Data/dlc.csv', 
                             delimiter=',', names=True, dtype=None, encoding=None)
    
        # 轉為 2D 陣列
        data_array = raw_data.view((np.float64, len(raw_data.dtype.names)))
    
        time_data = data_array[:, 0]  # 第一欄是時間
        state_data = data_array[:, 1:7].copy() # 取得 [vx, vy, vz, ax, ay, az]
    
        # 因為 raw data 是差分，這裡進行累加還原
        # 注意：這裡還原的是感測器的觀測值 (z)
        for i in range(1, state_data.shape[0]):
            state_data[i] += state_data[i - 1]
        
        return time_data, state_data

    @staticmethod
    def load_gt():
        raw_data = np.genfromtxt('./download/Flight1_Catered_Dataset-20201013/Data/truth.csv', 
                             delimiter=',', names=True, dtype=None, encoding=None)
    
        # 轉為 2D 陣列
        data_array = raw_data.view((np.float64, len(raw_data.dtype.names)))
    
        time_data = data_array[:, 0]  # 第一欄是時間
        state_data = data_array[:, 1:].copy() # 取得 [vx, vy, vz, ax, ay, az]
        return time_data, state_data

    @staticmethod
    def load_all_data():
        dlc_time_data, dlc_z_measurements = load_data.load_dlc()
        gt_time_data, gt_z_measurements = load_data.load_gt()
        return dlc_time_data, dlc_z_measurements, gt_time_data, gt_z_measurements

def run_kalman_filter(time_data, z_measurements, gt_pos, gt_vel, gt_quat):
    
    # --- 1. 初始化 Kalman Filter ---
    # dim_x=15: [px,py,pz, vx,vy,vz, ax,ay,az, angx,angy,angz, wx,wy,wz]
    # dim_z=6 : [vx, vy, vz, ax, ay, az]
    f = KalmanFilter(dim_x=15, dim_z=6)
    
    # 初始狀態: 用 truth 第一筆對齊原點與速度
    f.x = np.zeros(15)
    f.x[0:3] = gt_pos[0]
    f.x[3:6] = gt_vel[0]
    
    # 觀測矩陣 H (將 15 維狀態映射到 6 維觀測值)
    # 我們觀測到的是索引 3~5 (v) 和 6~8 (a)
    f.H = np.zeros((6, 15))
    f.H[0:3, 3:6] = np.eye(3) # 觀測到速度
    #f.H[3:6, 6:9] = np.eye(3) # 觀測到加速度
    f.H[3:6, 9:12] = np.eye(3)
    # 測量雜訊 R (假設感測器有一定的誤差)
    f.R = np.eye(6) * 0.1
    
    # 過程雜訊 Q (對狀態預測的不確定性)
    f.Q = np.eye(15) * 0.01
    
    # 初始協方差 P
    f.P *= 10.0

    # 用於儲存結果
    results = []

    # --- 2. 迭代執行 ---
    for i in range(1, len(time_data)):
        # 計算時間差 dt (單位通常需轉換為秒)
        dt = (time_data[i] - time_data[i-1]) / 1e9 # 假設原始數據是 ns
        
        if dt <= 0: continue # 防止異常數據

        # 更新狀態轉移矩陣 F (15x15)
        # 基於運動學: p_new = p + v*dt + 0.5*a*dt^2, v_new = v + a*dt
        f.F = np.eye(15)
        for j in range(3):
            # 位置部分
            f.F[j, j+3] = dt        # p = p + v*dt
            f.F[j, j+6] = 0.5*(dt**2) # p = p + 0.5*a*dt^2
            # 速度部分
            f.F[j+3, j+6] = dt      # v = v + a*dt
            # 角度部分
            f.F[j+9, j+12] = dt     # angle = angle + omega*dt

        # 預測
        f.predict()
        
        # 更新 (放入目前的觀測值 [vx, vy, vz, ax, ay, az])
        f.update(z_measurements[i])
        
        # 存入結果 (例如存入估算出的位置 px, py, pz)
        results.append(f.x.copy())

    results = np.array(results)
    # KF 內部估的是 IMU 位置，轉成 CON 位置再輸出
    final_pos_imu = results[-1, 0:3]
    final_pos_con = imu_pos_to_con_pos_ecef(final_pos_imu, gt_quat[-1])
    print("濾波完成，最終 IMU 位置(ECEF):", final_pos_imu)
    print("濾波完成，最終 CON 位置(ECEF):", final_pos_con)
    return results

def main():
    # 對齊 DLC 與 truth，並修正座標
    aligned = align_and_transform(
        "./download/Flight1_Catered_Dataset-20201013/Data/dlc.csv",
        "./download/Flight1_Catered_Dataset-20201013/Data/truth.csv",
    )

    time_ns = aligned["time_ns"]
    dv_imu = aligned["dv_imu"]
    da_imu = aligned["da_imu"]
    gt_pos = aligned["pos_ecef_sync"]
    gt_vel = aligned["vel_ecef_sync"]
    gt_quat = aligned["quat_con2ecef_sync"]

    # 觀測值 z: [vx, vy, vz, ax, ay, az]
    # 速度使用 truth 同步值，角速度/角加速度先用 IMU 原始值（保持維度一致）
    z_measurements = np.zeros((len(time_ns), 6))
    z_measurements[:, 0:3] = gt_vel
    z_measurements[:, 3:6] = da_imu

    run_kalman_filter(time_ns, z_measurements, gt_pos, gt_vel, gt_quat)

if __name__ == "__main__":
    main()
