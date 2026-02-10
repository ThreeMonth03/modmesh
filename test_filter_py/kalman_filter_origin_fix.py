import argparse
import numpy as np

from align_data import (
    POS_IMU_FROM_CON_IN_CON,
    align_and_transform,
    imu_pos_to_con_pos_ecef,
    imu_vec_to_ecef_variant,
    quat_to_dcm_con2ecef,
)


def con_pos_to_imu_pos_ecef(pos_con_ecef, quat_con2ecef):
    r_ecef_con = quat_to_dcm_con2ecef(quat_con2ecef)
    imu_offset_ecef = r_ecef_con @ POS_IMU_FROM_CON_IN_CON
    return pos_con_ecef + imu_offset_ecef


def dead_reckon_with_origin(aligned, variant):
    time_ns = aligned["time_ns"]
    dv_imu = aligned["dv_imu"]
    gt_pos = aligned["pos_ecef_sync"]
    gt_vel = aligned["vel_ecef_sync"]
    gt_quat = aligned["quat_con2ecef_sync"]

    pos_imu = con_pos_to_imu_pos_ecef(gt_pos[0], gt_quat[0])
    vel = gt_vel[0].copy()

    for i in range(1, len(time_ns)):
        dt = (time_ns[i] - time_ns[i - 1]) / 1e9
        if dt <= 0:
            continue
        dv_ecef = imu_vec_to_ecef_variant(dv_imu[i], gt_quat[i], variant)
        vel = vel + dv_ecef
        pos_imu = pos_imu + vel * dt

    pos_con = imu_pos_to_con_pos_ecef(pos_imu, gt_quat[-1])
    err = np.linalg.norm(pos_con - gt_pos[-1])
    return pos_imu, pos_con, err


def main():
    ap = argparse.ArgumentParser(
        description="Origin-corrected dead-reckoning using truth for ECEF origin."
    )
    ap.add_argument("--dlc", default="./download/Flight1_Catered_Dataset-20201013/Data/dlc.csv")
    ap.add_argument("--truth", default="./download/Flight1_Catered_Dataset-20201013/Data/truth.csv")
    ap.add_argument(
        "--target-time",
        default="dlc",
        choices=["dlc", "truth"],
        help="Timebase for aligned data.",
    )
    ap.add_argument("--variant", default="A", choices=["A", "B", "C"])
    args = ap.parse_args()

    aligned = align_and_transform(args.dlc, args.truth, target_time=args.target_time)
    pos_imu, pos_con, err = dead_reckon_with_origin(aligned, args.variant)

    print("Origin-corrected dead-reckoning result:")
    print(f"Variant {args.variant}: final_pos_imu_ecef = {pos_imu}")
    print(f"Variant {args.variant}: final_pos_con_ecef = {pos_con}")
    print(f"Variant {args.variant}: final_error_m = {err}")


if __name__ == "__main__":
    main()
