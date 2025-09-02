# check_dataset_SE3_new.py
import os
import numpy as np

np.set_printoptions(precision=6, suppress=True)

# ---------- helpers ----------
def decompose_se3(T4):
    R = T4[:3, :3]
    p = T4[:3, 3]
    return R, p

def rot_angle(R_err):
    tr = float(np.trace(R_err))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))  # numerical safety
    return float(np.arccos(c))

def print_T(name, T):
    print(f"{name}:\n{T}")

# ---------- I/O ----------
link_count = int(input("How many links do you want to check?: ").strip())
data_dir = os.path.join("data_storage", f"link_{link_count}")
data_path = os.path.join(data_dir, "fault_dataset.npz")

if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found.")

data = np.load(data_path)

# ====== Keys ======
has_rel = ('desired_link_rel' in data.files) and ('actual_link_rel' in data.files)
has_cum = ('desired_link_cum' in data.files) and ('actual_link_cum' in data.files)

if not has_cum:
    raise KeyError("Dataset must include desired_link_cum / actual_link_cum to reconstruct EE.")

desired_link_rel = data['desired_link_rel'] if has_rel else None
actual_link_rel  = data['actual_link_rel']  if has_rel else None
desired_link_cum = data['desired_link_cum']
actual_link_cum  = data['actual_link_cum']

# EE = 마지막 누적변환 (T_{0->N})
desired_ee = desired_link_cum[:, :, -1]  # (S,T,4,4)
actual_ee  = actual_link_cum[:, :, -1]

# required label
label = data['label']  # (S,T,8*link_count)

# meta
if 'dof' in data.files:
    dof = int(data['dof'])
else:
    dof = link_count

# ---------- Shapes / format ----------
S, T, _, _ = desired_ee.shape
motor_count = label.shape[2]

fmt_parts = ["EE(derived from T_{0->N})"]
if has_rel: fmt_parts.append("PER_LINK_REL")
if has_cum: fmt_parts.append("PER_LINK_CUM")
fmt = " + ".join(fmt_parts)

print(f"Format: {fmt}")
print(f"desired_ee.shape   = {desired_ee.shape}")
print(f"actual_ee.shape    = {actual_ee.shape}")
if has_rel:
    print(f"desired_link_rel   = {desired_link_rel.shape}  # T_(i->i+1)")
    print(f"actual_link_rel    = {actual_link_rel.shape}")
print(f"desired_link_cum   = {desired_link_cum.shape}  # T_(0->k+1)")
print(f"actual_link_cum    = {actual_link_cum.shape}")
print(f"label.shape        = {label.shape} (motor_count={motor_count})")
print(f"link_count (N)     = {link_count}")
print(f"dof (total joints) = {dof}")

# ---------- Time indices ----------
raw_times = [0, 1, 10, 50, 100, 150, 200, 250, 500, 750, 800, 999]
time_idx_list = [t for t in raw_times if t < T]

# ---------- Which sample to inspect ----------
sample_idx = 8
assert 0 <= sample_idx < S, "sample_idx out of range"

# ---------- Inspection ----------
for time_idx in time_idx_list:
    print("\n" + "="*50)
    print(f"[Sample {sample_idx}, Time {time_idx}]")

    lbl = label[sample_idx, time_idx].astype(int)
    faulty_idx = np.where(lbl == 0)[0]
    print("Motor labels:", lbl)
    if faulty_idx.size > 0:
        print(f"→ Faulty motors at this time: {faulty_idx.tolist()}")

    # --- EE = 마지막 누적변환 ---
    T_des_ee = desired_ee[sample_idx, time_idx]
    T_act_ee = actual_ee[sample_idx, time_idx]
    R_des_ee, p_des_ee = decompose_se3(T_des_ee)
    R_act_ee, p_act_ee = decompose_se3(T_act_ee)

    print("\n[EE=T_{0->N}] Desired:")
    print(T_des_ee)
    print("\n[EE=T_{0->N}] Actual :")
    print(T_act_ee)

    ee_pos_err = np.linalg.norm(p_act_ee - p_des_ee)
    R_err_ee   = R_des_ee.T @ R_act_ee
    ee_ang_err = rot_angle(R_err_ee)
    ee_ang_deg = np.degrees(ee_ang_err)

    print(f"\n[EE=T0N] Position error = {ee_pos_err:.6f} m")
    print(f"[EE=T0N] Orientation error = {ee_ang_err:.6f} rad ({ee_ang_deg:.3f} deg)")

    # --- Per-link relative transforms ---
    if has_rel:
        per_link_pos = []
        per_link_ang = []

        print("\nPer-link relative transforms & errors (T_{i-1,i}):")
        for i in range(link_count):
            T_rel_des = desired_link_rel[sample_idx, time_idx, i]
            T_rel_act = actual_link_rel[sample_idx, time_idx, i]
            R_rel_des, p_rel_des = decompose_se3(T_rel_des)
            R_rel_act, p_rel_act = decompose_se3(T_rel_act)

            print(f"\n  - T_{{{i}->{i+1}}} (Desired):")
            print(T_rel_des)
            print(f"  - T_{{{i}->{i+1}}} (Actual) :")
            print(T_rel_act)

            j_pos = np.linalg.norm(p_rel_act - p_rel_des)
            Rj_err = R_rel_des.T @ R_rel_act
            j_ang = rot_angle(Rj_err)
            j_deg = np.degrees(j_ang)

            per_link_pos.append(j_pos)
            per_link_ang.append(j_ang)

            print(f"    -> pos_err={j_pos:.6f} m, ori_err={j_ang:.6f} rad ({j_deg:.3f} deg)")

        per_link_pos = np.array(per_link_pos)
        per_link_ang = np.array(per_link_ang)
        j_max_pos = int(np.argmax(per_link_pos))
        j_max_ang = int(np.argmax(per_link_ang))

        print("\nSummary at this time (per-link relative):")
        print(f"  mean pos_err = {per_link_pos.mean():.6f} m | "
              f"max={per_link_pos.max():.6f} m @ joint {j_max_pos+1}")
        print(f"  mean ori_err = {per_link_ang.mean():.6f} rad "
              f"({np.degrees(per_link_ang).mean():.3f} deg) | "
              f"max={per_link_ang.max():.6f} rad "
              f"({np.degrees(per_link_ang).max():.3f} deg) @ joint {j_max_ang+1}")
