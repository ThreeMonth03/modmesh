import argparse
import numpy as np

from align_data import (
    POS_IMU_FROM_CON_IN_CON,
    DCM_CON_IMU,
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


def gravity_ecef(pos_ecef):
    r = np.linalg.norm(pos_ecef)
    if r == 0.0:
        return np.zeros(3)
    return -MU_EARTH / (r**3) * pos_ecef


def gravity_ecef_wgs84(pos_ecef):
    x, y, z = pos_ecef
    r = np.linalg.norm(pos_ecef)
    if r == 0.0:
        return np.zeros(3)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, np.sqrt(x * x + y * y))
    sin_lat = np.sin(lat)
    gamma = WGS84_GAMMA_E * (1.0 + WGS84_K * sin_lat * sin_lat) / np.sqrt(
        1.0 - WGS84_E2 * sin_lat * sin_lat
    )
    g_mag = gamma
    # Approximate gravity along the radial direction in ECEF
    return -g_mag * (pos_ecef / r)


def mechanize(aligned, variant="A", use_earth_rotation=True):
    time_ns = aligned["time_ns"]
    dv_imu = aligned["dv_imu"]
    da_imu = aligned["da_imu"]
    gt_pos = aligned["pos_ecef_sync"]
    gt_vel = aligned["vel_ecef_sync"]
    gt_quat = aligned["quat_con2ecef_sync"]

    pos_con = gt_pos[0].copy()
    vel_con = gt_vel[0].copy()

    dcm_ecef_con = quat_to_dcm_con2ecef(gt_quat[0])
    if variant == "A":
        dcm_con_imu = DCM_CON_IMU
    elif variant == "B":
        dcm_con_imu = DCM_CON_IMU.T
    elif variant == "C":
        # Use inverse of CON->ECEF (i.e., ECEF->CON) with nominal IMU alignment
        dcm_con_imu = DCM_CON_IMU
        dcm_ecef_con = dcm_ecef_con.T
    elif variant == "D":
        # Variant C direction, but with WGS84 normal gravity model
        dcm_con_imu = DCM_CON_IMU
        dcm_ecef_con = dcm_ecef_con.T
    else:
        raise ValueError(f"Unknown variant: {variant}")

    dcm_ecef_imu = dcm_ecef_con @ dcm_con_imu

    omega_e = np.array([0.0, 0.0, OMEGA_EARTH])

    for i in range(1, len(time_ns)):
        dt = (time_ns[i] - time_ns[i - 1]) / 1e9
        if dt <= 0:
            continue

        # Update attitude using IMU delta-angle (body-frame)
        dcm_delta = dcm_from_delta_angle(da_imu[i])
        dcm_ecef_imu = dcm_ecef_imu @ dcm_delta

        # Transform delta-velocity to ECEF
        dv_ecef = dcm_ecef_imu @ dv_imu[i]
        accel_ecef = dv_ecef / dt

        # Gravity and Earth rotation compensation
        if variant == "D":
            accel_ecef = accel_ecef + gravity_ecef_wgs84(pos_con)
        else:
            accel_ecef = accel_ecef + gravity_ecef(pos_con)
        if use_earth_rotation:
            accel_ecef = accel_ecef - 2.0 * np.cross(omega_e, vel_con)
            accel_ecef = accel_ecef - np.cross(omega_e, np.cross(omega_e, pos_con))

        vel_con = vel_con + accel_ecef * dt
        pos_con = pos_con + vel_con * dt

    return pos_con, vel_con


def main():
    ap = argparse.ArgumentParser(description="Simple strapdown INS mechanization from IMU dv/da.")
    ap.add_argument("--dlc", default="./download/Flight1_Catered_Dataset-20201013/Data/dlc.csv")
    ap.add_argument("--truth", default="./download/Flight1_Catered_Dataset-20201013/Data/truth.csv")
    ap.add_argument(
        "--target-time",
        default="dlc",
        choices=["dlc", "truth"],
        help="Timebase for aligned data.",
    )
    ap.add_argument("--variant", default="C", choices=["A", "B", "C", "D"], type=str.upper)
    ap.add_argument("--no-earth-rotation", action="store_true")
    args = ap.parse_args()

    aligned = align_and_transform(args.dlc, args.truth, target_time=args.target_time)
    pos_con, vel_con = mechanize(
        aligned,
        variant=args.variant,
        use_earth_rotation=not args.no_earth_rotation,
    )

    gt_pos = aligned["pos_ecef_sync"]
    err = np.linalg.norm(pos_con - gt_pos[-1])

    print("INS mechanization result:")
    print(f"Variant {args.variant}: final_pos_con_ecef = {pos_con}")
    print(f"Variant {args.variant}: final_error_m = {err}")


if __name__ == "__main__":
    main()
