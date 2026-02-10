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
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_GAMMA_E = 9.7803253359
WGS84_K = 0.00193185265241


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


def gravity_ecef_point_mass(pos_ecef):
    r = np.linalg.norm(pos_ecef)
    if r == 0.0:
        return np.zeros(3)
    return -MU_EARTH / (r**3) * pos_ecef


def gravity_ecef_wgs84(pos_ecef):
    x, y, z = pos_ecef
    r = np.linalg.norm(pos_ecef)
    if r == 0.0:
        return np.zeros(3)
    lat = np.arctan2(z, np.sqrt(x * x + y * y))
    sin_lat = np.sin(lat)
    gamma = WGS84_GAMMA_E * (1.0 + WGS84_K * sin_lat * sin_lat) / np.sqrt(
        1.0 - WGS84_E2 * sin_lat * sin_lat
    )
    return -gamma * (pos_ecef / r)


def con_pos_to_imu_pos_ecef(pos_con_ecef, dcm_ecef_con):
    imu_offset_ecef = dcm_ecef_con @ POS_IMU_FROM_CON_IN_CON
    return pos_con_ecef + imu_offset_ecef


def imu_pos_to_con_pos_ecef(pos_imu_ecef, dcm_ecef_con):
    imu_offset_ecef = dcm_ecef_con @ POS_IMU_FROM_CON_IN_CON
    return pos_imu_ecef - imu_offset_ecef


def run_ins_kf_bias(
    aligned,
    variant="C",
    use_earth_rotation=True,
    update_every=0,
):
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

    # KF state: [px, py, pz, vx, vy, vz, bax, bay, baz] in ECEF (IMU position)
    kf = KalmanFilter(dim_x=9, dim_z=6)
    pos_imu0 = con_pos_to_imu_pos_ecef(gt_pos[0], dcm_ecef_con)
    kf.x = np.hstack([pos_imu0, gt_vel[0], np.zeros(3)])
    kf.P *= 1.0
    kf.Q = np.eye(9) * 0.01
    kf.R = np.eye(6) * 0.01

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
        if variant == "D":
            accel_ecef = accel_ecef + gravity_ecef_wgs84(pos_imu)
        else:
            accel_ecef = accel_ecef + gravity_ecef_point_mass(pos_imu)
        if use_earth_rotation:
            accel_ecef = accel_ecef - 2.0 * np.cross(omega_e, vel)
            accel_ecef = accel_ecef - np.cross(omega_e, np.cross(omega_e, pos_imu))

        # Subtract accel bias state
        accel_ecef = accel_ecef - kf.x[6:9]

        # KF prediction: x = F x + B u
        kf.F = np.eye(9)
        kf.F[0:3, 3:6] = np.eye(3) * dt
        B = np.zeros((9, 3))
        B[0:3, 0:3] = 0.5 * dt**2 * np.eye(3)
        B[3:6, 0:3] = dt * np.eye(3)
        kf.predict(u=accel_ecef, B=B)

        # Optional truth update (sparse)
        if update_every and (i % update_every == 0):
            kf.H = np.zeros((6, 9))
            kf.H[0:3, 0:3] = np.eye(3)
            kf.H[3:6, 3:6] = np.eye(3)
            z = np.hstack([gt_pos[i], gt_vel[i]])
            kf.update(z)

    final_pos_imu = kf.x[0:3]
    final_pos_con = imu_pos_to_con_pos_ecef(final_pos_imu, dcm_ecef_con)
    err = np.linalg.norm(final_pos_con - gt_pos[-1])
    return final_pos_imu, final_pos_con, err


def main():
    ap = argparse.ArgumentParser(
        description="INS-KF with accel bias (Variant C direction, optional WGS84 gravity)."
    )
    ap.add_argument("--dlc", default="./download/Flight1_Catered_Dataset-20201013/Data/dlc.csv")
    ap.add_argument("--truth", default="./download/Flight1_Catered_Dataset-20201013/Data/truth.csv")
    ap.add_argument(
        "--target-time",
        default="dlc",
        choices=["dlc", "truth"],
        help="Timebase for aligned data.",
    )
    ap.add_argument("--variant", default="C", choices=["C", "D"], type=str.upper)
    ap.add_argument("--no-earth-rotation", action="store_true")
    ap.add_argument(
        "--update-every",
        type=int,
        default=0,
        help="Use truth update every N steps (0 = no update)",
    )
    args = ap.parse_args()

    aligned = align_and_transform(args.dlc, args.truth, target_time=args.target_time)
    pos_imu, pos_con, err = run_ins_kf_bias(
        aligned,
        variant=args.variant,
        use_earth_rotation=not args.no_earth_rotation,
        update_every=args.update_every,
    )

    print("INS-KF with accel bias result:")
    print(f"variant = {args.variant}, update_every = {args.update_every}")
    print(f"final_pos_imu_ecef = {pos_imu}")
    print(f"final_pos_con_ecef = {pos_con}")
    print(f"final_error_m = {err}")


if __name__ == "__main__":
    main()
