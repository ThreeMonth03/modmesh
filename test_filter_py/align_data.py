import argparse
import numpy as np

# Constants from download/Flight1_Catered_Dataset-20201013/README.txt
POS_IMU_FROM_CON_IN_CON = np.array([-0.08035, 0.28390, -1.42333])
DCM_CON_IMU = np.array([
    [-0.2477, -0.1673,  0.9543],
    [-0.0478,  0.9859,  0.1604],
    [-0.9677, -0.0059, -0.2522],
])


def _genfromtxt_2d(path):
    raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    return raw.view((np.float64, len(raw.dtype.names)))


def load_dlc(path):
    data = _genfromtxt_2d(path)
    time_ns = data[:, 0]
    dv_imu = data[:, 1:4]
    da_imu = data[:, 4:7]
    return time_ns, dv_imu, da_imu


def load_truth(path):
    data = _genfromtxt_2d(path)
    time_ns = data[:, 0]
    pos_ecef = data[:, 1:4]
    vel_ecef = data[:, 4:7]
    quat_con2ecef = data[:, 7:11]
    return time_ns, pos_ecef, vel_ecef, quat_con2ecef


def interp_linear(t_src, y_src, t_tgt):
    if y_src.ndim == 1:
        return np.interp(t_tgt, t_src, y_src)
    out = np.empty((len(t_tgt), y_src.shape[1]), dtype=np.float64)
    for i in range(y_src.shape[1]):
        out[:, i] = np.interp(t_tgt, t_src, y_src[:, i])
    return out


def interp_quat_nlerp(t_src, q_src, t_tgt):
    out = np.empty((len(t_tgt), 4), dtype=np.float64)
    j = 0
    n = len(t_src)
    for i, t in enumerate(t_tgt):
        while j + 1 < n and t_src[j + 1] < t:
            j += 1
        if j + 1 >= n:
            out[i] = q_src[-1]
            continue
        t0 = t_src[j]
        t1 = t_src[j + 1]
        if t1 == t0:
            out[i] = q_src[j]
            continue
        w = (t - t0) / (t1 - t0)
        q0 = q_src[j]
        q1 = q_src[j + 1]
        if np.dot(q0, q1) < 0.0:
            q1 = -q1
        qi = (1.0 - w) * q0 + w * q1
        norm = np.linalg.norm(qi)
        if norm == 0.0:
            out[i] = q0
        else:
            out[i] = qi / norm
    return out


def resample_delta_via_cumulative(t_src, delta_src, t_tgt):
    # Delta measurements (e.g. delta-v, delta-angle) are interval increments.
    # Resample by interpolating cumulative integral, then differencing on target grid.
    if delta_src.ndim == 1:
        delta_eff = delta_src.copy()
        if len(delta_eff) > 0:
            delta_eff[0] = 0.0
        cum_src = np.cumsum(delta_eff)
        cum_tgt = np.interp(t_tgt, t_src, cum_src)
        out = np.empty(len(t_tgt), dtype=np.float64)
        if len(out) > 0:
            out[0] = 0.0
        if len(out) > 1:
            out[1:] = cum_tgt[1:] - cum_tgt[:-1]
        return out

    delta_eff = delta_src.copy()
    if len(delta_eff) > 0:
        delta_eff[0, :] = 0.0
    cum_src = np.cumsum(delta_eff, axis=0)
    cum_tgt = interp_linear(t_src, cum_src, t_tgt)
    out = np.empty((len(t_tgt), delta_src.shape[1]), dtype=np.float64)
    if len(out) > 0:
        out[0, :] = 0.0
    if len(out) > 1:
        out[1:, :] = cum_tgt[1:, :] - cum_tgt[:-1, :]
    return out


def quat_to_dcm_con2ecef(q):
    x, y, z, w = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ])


def imu_vec_to_ecef(vec_imu, quat_con2ecef):
    vec_con = DCM_CON_IMU @ vec_imu
    r_ecef_con = quat_to_dcm_con2ecef(quat_con2ecef)
    return r_ecef_con @ vec_con


def imu_vec_to_ecef_variant(vec_imu, quat_con2ecef, variant):
    if variant == "A":
        vec_con = DCM_CON_IMU @ vec_imu
        r_ecef_con = quat_to_dcm_con2ecef(quat_con2ecef)
        return r_ecef_con @ vec_con
    if variant == "B":
        vec_con = DCM_CON_IMU.T @ vec_imu
        r_ecef_con = quat_to_dcm_con2ecef(quat_con2ecef)
        return r_ecef_con @ vec_con
    if variant == "C":
        vec_con = DCM_CON_IMU @ vec_imu
        r_ecef_con = quat_to_dcm_con2ecef(quat_con2ecef).T
        return r_ecef_con @ vec_con
    if variant == "D":
        vec_con = DCM_CON_IMU.T @ vec_imu
        r_ecef_con = quat_to_dcm_con2ecef(quat_con2ecef).T
        return r_ecef_con @ vec_con
    raise ValueError(f"Unknown variant: {variant}")


def imu_pos_to_con_pos_ecef(pos_imu_ecef, quat_con2ecef):
    r_ecef_con = quat_to_dcm_con2ecef(quat_con2ecef)
    imu_offset_ecef = r_ecef_con @ POS_IMU_FROM_CON_IN_CON
    return pos_imu_ecef - imu_offset_ecef


def align_and_transform(dlc_path, truth_path, target_time="dlc"):
    dlc_time, dv_imu, da_imu = load_dlc(dlc_path)
    gt_time, pos_ecef, vel_ecef, quat_con2ecef = load_truth(truth_path)

    if target_time == "dlc":
        time_ns = dlc_time
        dv_imu_sync = dv_imu
        da_imu_sync = da_imu
        pos_ecef_sync = interp_linear(gt_time, pos_ecef, time_ns)
        vel_ecef_sync = interp_linear(gt_time, vel_ecef, time_ns)
        quat_sync = interp_quat_nlerp(gt_time, quat_con2ecef, time_ns)
    elif target_time == "truth":
        time_ns = gt_time
        dv_imu_sync = resample_delta_via_cumulative(dlc_time, dv_imu, time_ns)
        da_imu_sync = resample_delta_via_cumulative(dlc_time, da_imu, time_ns)
        pos_ecef_sync = pos_ecef.copy()
        vel_ecef_sync = vel_ecef.copy()
        quat_sync = quat_con2ecef.copy()
    else:
        raise ValueError("target_time must be either 'dlc' or 'truth'")

    dv_ecef = np.empty_like(dv_imu_sync)
    da_con = np.empty_like(da_imu_sync)
    for i in range(len(time_ns)):
        dv_ecef[i] = imu_vec_to_ecef(dv_imu_sync[i], quat_sync[i])
        da_con[i] = DCM_CON_IMU @ da_imu_sync[i]

    return {
        "time_ns": time_ns,
        "dv_imu": dv_imu_sync,
        "da_imu": da_imu_sync,
        "dv_ecef": dv_ecef,
        "da_con": da_con,
        "pos_ecef_sync": pos_ecef_sync,
        "vel_ecef_sync": vel_ecef_sync,
        "quat_con2ecef_sync": quat_sync,
    }


def run_experiments(dlc_path, truth_path, target_time="dlc"):
    aligned = align_and_transform(dlc_path, truth_path, target_time=target_time)
    time_ns = aligned["time_ns"]
    dv_imu = aligned["dv_imu"]
    pos_gt = aligned["pos_ecef_sync"]
    vel_gt = aligned["vel_ecef_sync"]
    quat_sync = aligned["quat_con2ecef_sync"]

    results = {}
    for variant in ["A", "B", "C", "D"]:
        pos = pos_gt[0].copy()
        vel = vel_gt[0].copy()

        for i in range(1, len(time_ns)):
            dt = (time_ns[i] - time_ns[i - 1]) / 1e9
            if dt <= 0:
                continue
            dv_ecef = imu_vec_to_ecef_variant(dv_imu[i], quat_sync[i], variant)
            vel = vel + dv_ecef
            pos = pos + vel * dt

        err = np.linalg.norm(pos - pos_gt[-1])
        results[variant] = (pos, err)

    print("Experiment summary (dead-reckoning from IMU dv):")
    for variant in ["A", "B", "C", "D"]:
        pos, err = results[variant]
        print(f"Variant {variant}: final_pos_ecef = {pos}, final_error_m = {err}")


def main():
    ap = argparse.ArgumentParser(description="Align DLC IMU data to truth and provide correct frame transforms.")
    ap.add_argument("--dlc", default="./download/Flight1_Catered_Dataset-20201013/Data/dlc.csv")
    ap.add_argument("--truth", default="./download/Flight1_Catered_Dataset-20201013/Data/truth.csv")
    ap.add_argument(
        "--target-time",
        default="dlc",
        choices=["dlc", "truth"],
        help="Timebase for aligned output: keep DLC samples, or interpolate DLC to truth timestamps.",
    )
    ap.add_argument("--out", default="./test_filter_py/aligned_data.npz")
    ap.add_argument("--experiments", action="store_true", help="Run frame-transform experiments and print final results")
    args = ap.parse_args()

    aligned = align_and_transform(args.dlc, args.truth, target_time=args.target_time)
    np.savez(args.out, **aligned)

    if args.experiments:
        run_experiments(args.dlc, args.truth, target_time=args.target_time)

    # Example usage note for KF position transform:
    # If your Kalman filter estimates IMU position in ECEF, convert to CON position in ECEF via:
    # pos_con_ecef = imu_pos_to_con_pos_ecef(pos_imu_ecef, quat_con2ecef)
    # where quat_con2ecef should be aligned to the same timestamp.


if __name__ == "__main__":
    main()
