import argparse

from align_data import align_and_transform
from kalman_filter_ins_bias import run_ins_kf_bias


def main():
    ap = argparse.ArgumentParser(
        description="Align DLC/truth data on selected timebase, then run INS-KF fusion."
    )
    ap.add_argument("--dlc", default="./download/Flight1_Catered_Dataset-20201013/Data/dlc.csv")
    ap.add_argument("--truth", default="./download/Flight1_Catered_Dataset-20201013/Data/truth.csv")
    ap.add_argument(
        "--target-time",
        default="dlc",
        choices=["dlc", "truth"],
        help="Timebase for aligned data before fusion.",
    )
    ap.add_argument("--variant", default="C", choices=["C", "D"], type=str.upper)
    ap.add_argument(
        "--update-every",
        type=int,
        default=100,
        help="Use truth update every N steps (0 = no update)",
    )
    ap.add_argument("--no-earth-rotation", action="store_true")
    args = ap.parse_args()

    aligned = align_and_transform(args.dlc, args.truth, target_time=args.target_time)
    pos_imu, pos_con, err = run_ins_kf_bias(
        aligned,
        variant=args.variant,
        use_earth_rotation=not args.no_earth_rotation,
        update_every=args.update_every,
    )

    print("Fusion result (aligned on selected timebase):")
    print(f"variant = {args.variant}, update_every = {args.update_every}")
    print(f"final_pos_imu_ecef = {pos_imu}")
    print(f"final_pos_con_ecef = {pos_con}")
    print(f"final_error_m = {err}")


if __name__ == "__main__":
    main()
