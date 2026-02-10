import argparse
import numpy as np
from filterpy.kalman import KalmanFilter

from align_data import (
    DCM_CON_IMU,
    POS_IMU_FROM_CON_IN_CON,
    align_and_transform,
    quat_to_dcm_con2ecef,
)

OMEGA_EARTH = 7.292115e-5  # rad/s
MU_EARTH = 3.986004418e14  # m^3/s^2


def skew(v):
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def dcm_from_delta_angle(dtheta):
    angle = np.linalg.norm(dtheta)
    if angle < 1e-12:
        return np.eye(3) + skew(dtheta)
    k = dtheta / angle
    K = skew(k)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def gravity_ecef(pos_ecef):
    r = np.linalg.norm(pos_ecef)
    if r == 0.0:
        return np.zeros(3)
    return -MU_EARTH / (r**3) * pos_ecef


def con_pos_to_imu_pos_ecef(pos_con_ecef, dcm_ecef_con):
    imu_offset_ecef = dcm_ecef_con @ POS_IMU_FROM_CON_IN_CON
    return pos_con_ecef + imu_offset_ecef


def imu_pos_to_con_pos_ecef(pos_imu_ecef, dcm_ecef_con):
    imu_offset_ecef = dcm_ecef_con @ POS_IMU_FROM_CON_IN_CON
    return pos_imu_ecef - imu_offset_ecef


def run_ins_kf(aligned, use_earth_rotation=True):
    time_ns = aligned["time_ns"]
    dv_imu = aligned["dv_imu"]
    da_imu = aligned["da_imu"]
    gt_pos = aligned["pos_ecef_sync"]
    gt_vel = aligned["vel_ecef_sync"]
    gt_quat = aligned["quat_con2ecef_sync"]

    # Variant C direction: use inverse of CON->ECEF
    dcm_ecef_con = quat_to_dcm_con2ecef(gt_quat[0]).T
    dcm_con_imu = DCM_CON_IMU
    dcm_ecef_imu = dcm_ecef_con @ dcm_con_imu

    # KF state: [px, py, pz, vx, vy, vz] in ECEF (IMU position)
    # filterpy requires dim_z >= 1 even if we never call update()
    kf = KalmanFilter(dim_x=6, dim_z=1)
    pos_imu0 = con_pos_to_imu_pos_ecef(gt_pos[0], dcm_ecef_con)
    kf.x = np.hstack([pos_imu0, gt_vel[0]])
    print("init gt_pos_ecef =", gt_pos[0])
    print("init gt_vel_ecef =", gt_vel[0])
    print("init imu_pos_ecef =", pos_imu0)
    kf.P *= 1.0
    kf.Q = np.eye(6) * 0.01

    omega_e = np.array([0.0, 0.0, OMEGA_EARTH])

    for i in range(1, len(time_ns)):
        dt = (time_ns[i] - time_ns[i - 1]) / 1e9
        if dt <= 0:
            continue

        # Update attitude using IMU delta-angle
        dcm_delta = dcm_from_delta_angle(da_imu[i])
        dcm_ecef_imu = dcm_ecef_imu @ dcm_delta

        # Transform delta-velocity to ECEF
        dv_ecef = dcm_ecef_imu @ dv_imu[i]
        accel_ecef = dv_ecef / dt

        # Gravity and Earth rotation compensation (ECEF)
        pos_imu = kf.x[0:3]
        vel = kf.x[3:6]
        accel_ecef = accel_ecef + gravity_ecef(pos_imu)
        if use_earth_rotation:
            accel_ecef = accel_ecef - 2.0 * np.cross(omega_e, vel)
            accel_ecef = accel_ecef - np.cross(omega_e, np.cross(omega_e, pos_imu))

        # KF prediction: x = F x + B u
        kf.F = np.eye(6)
        kf.F[0:3, 3:6] = np.eye(3) * dt
        B = np.zeros((6, 3))
        B[0:3, 0:3] = 0.5 * dt**2 * np.eye(3)
        B[3:6, 0:3] = dt * np.eye(3)
        kf.predict(u=accel_ecef, B=B)

    final_pos_imu = kf.x[0:3]
    final_pos_con = imu_pos_to_con_pos_ecef(final_pos_imu, dcm_ecef_con)
    err = np.linalg.norm(final_pos_con - gt_pos[-1])
    return final_pos_imu, final_pos_con, err


def main():
    ap = argparse.ArgumentParser(
        description="INS-based KF prediction (Variant C direction, no truth update)."
    )
    ap.add_argument("--dlc", default="./download/Flight1_Catered_Dataset-20201013/Data/dlc.csv")
    ap.add_argument("--truth", default="./download/Flight1_Catered_Dataset-20201013/Data/truth.csv")
    ap.add_argument(
        "--target-time",
        default="dlc",
        choices=["dlc", "truth"],
        help="Timebase for aligned data.",
    )
    ap.add_argument("--no-earth-rotation", action="store_true")
    args = ap.parse_args()

    aligned = align_and_transform(args.dlc, args.truth, target_time=args.target_time)
    pos_imu, pos_con, err = run_ins_kf(
        aligned, use_earth_rotation=not args.no_earth_rotation
    )

    print("INS-KF prediction result (no truth update):")
    print(f"final_pos_imu_ecef = {pos_imu}")
    print(f"final_pos_con_ecef = {pos_con}")
    print(f"final_error_m = {err}")


if __name__ == "__main__":
    main()
