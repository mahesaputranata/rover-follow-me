#!/usr/bin/env python3
"""
summarize_run.py - summarize one recorded rover run (ros2 bag) into
summary.json + plots.png, with a small rulebook that flags known failure modes.

Usage:
    python summarize_run.py <run_dir> [--no-plot]

<run_dir> is a folder made by run_recorder.py:  <run_dir>/bag/ (rosbag2 sqlite3)
and <run_dir>/config/ (settings snapshot). Runs on the PC, no ROS needed:
    pip install rosbags numpy matplotlib

Topics used: /bbox_error /cmd_vel /rover/state /esp32/telemetry /esp32/warnings
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Zone thresholds mirrored from lqr_node.cpp bbox_callback (hysteresis).
FOLLOW_ENTER, FOLLOW_EXIT = 0.70, 0.80
REVERSE_ENTER, REVERSE_EXIT = 1.40, 1.25
SP_DEADBAND = 0.08          # ESP32 SP_DEADBAND (m/s)
PWM_SAT = 245               # |pwm| at/above this counts as saturated

PID_RE = re.compile(
    r"PID dt=(?P<dt>[-\d.]+) velA=(?P<velA>[-\d.]+) velB=(?P<velB>[-\d.]+) "
    r"spA=(?P<spA>[-\d.]+) spB=(?P<spB>[-\d.]+) "
    r"pwmA=(?P<pwmA>-?\d+) pwmB=(?P<pwmB>-?\d+)"
)


# --------------------------------------------------------------------------
#  loading
# --------------------------------------------------------------------------
def load_run(run_dir):
    bag = Path(run_dir) / "bag"
    rows = {"bbox": [], "cmd": [], "state": [], "pid": [], "warn": []}
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([bag], default_typestore=typestore) as reader:
        t0 = reader.start_time * 1e-9
        for conn, ts, raw in reader.messages():
            msg = reader.deserialize(raw, conn.msgtype)
            t = ts * 1e-9 - t0
            topic = conn.topic
            if topic == "/bbox_error":
                d = [float(x) for x in msg.data]
                if len(d) >= 4:
                    rows["bbox"].append((t, d[0], d[1], d[2], d[3]))
            elif topic == "/cmd_vel":
                rows["cmd"].append((t, msg.linear.x, msg.angular.z))
            elif topic == "/rover/state":
                rows["state"].append((t, msg.twist.twist.linear.x, msg.twist.twist.angular.z))
            elif topic == "/esp32/telemetry":
                m = PID_RE.search(msg.data)
                if m:
                    rows["pid"].append((
                        t, float(m["velA"]), float(m["velB"]),
                        float(m["spA"]), float(m["spB"]),
                        float(m["pwmA"]), float(m["pwmB"])))
            elif topic == "/esp32/warnings":
                rows["warn"].append((t, msg.data))
        duration = (reader.end_time - reader.start_time) * 1e-9
    data = {k: (np.array(v, dtype=float) if k != "warn" else v) for k, v in rows.items()}
    return data, duration


# --------------------------------------------------------------------------
#  helpers
# --------------------------------------------------------------------------
def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def runs_of(mask):
    """Return (start_idx, end_idx_inclusive) of contiguous True runs."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def zone_series(area, valid):
    """Replay the lqr_node hysteresis. 1 = follow, 0 = hold, -1 = reverse."""
    zone, out = 0, []
    for a, v in zip(area, valid):
        if v < 0.5:
            zone = 0
        else:
            if zone == 1 and a > FOLLOW_EXIT:
                zone = 0
            elif zone == -1 and a < REVERSE_EXIT:
                zone = 0
            if zone == 0:
                if a < FOLLOW_ENTER:
                    zone = 1
                elif a > REVERSE_ENTER:
                    zone = -1
        out.append(zone)
    return np.array(out)


def smooth_grid(cmd, dt=0.05, win_s=0.5):
    """Resample /cmd_vel to a uniform grid and smooth it (it is noisy at 45 Hz)."""
    t = cmd[:, 0]
    grid = np.arange(t[0], t[-1], dt)
    lin = np.interp(grid, t, cmd[:, 1])
    ang = np.interp(grid, t, cmd[:, 2])
    k = max(1, int(win_s / dt))
    kern = np.ones(k) / k
    return grid, np.convolve(lin, kern, "same"), np.convolve(ang, kern, "same")


# --------------------------------------------------------------------------
#  analysis
# --------------------------------------------------------------------------
def analyze(data, duration):
    s = {"duration_s": round(duration, 1)}
    flags = []

    def flag(level, code, msg):
        flags.append({"level": level, "code": code, "message": msg})

    bbox, cmd, state, pid = data["bbox"], data["cmd"], data["state"], data["pid"]

    # ---- vision ----------------------------------------------------------
    if len(bbox):
        t = bbox[:, 0]
        off, area, valid = bbox[:, 1], bbox[:, 3], bbox[:, 4]
        v = valid > 0.5
        s["bbox_msgs"] = int(len(bbox))

        # only the part of the run after the first valid sample counts as "tracking"
        first_valid = int(np.argmax(v)) if v.any() else None
        s["valid_fraction"] = round(float(v[first_valid:].mean()), 3) if first_valid is not None else 0.0
        s["valid_fraction_whole_run"] = round(float(v.mean()), 3)
        s["initial_wait_s"] = round(float(t[first_valid]), 1) if first_valid is not None else None

        gaps = []
        if first_valid is not None:
            for a, b in runs_of(~v):
                if a < first_valid:
                    continue
                end = t[b + 1] if b + 1 < len(t) else t[b]
                gaps.append({"start_s": round(float(t[a]), 1), "duration_s": round(float(end - t[a]), 1),
                             "to_end": bool(b + 1 >= len(t))})
        s["lost_gaps"] = [g for g in gaps if g["duration_s"] >= 1.0]
        s["longest_lost_gap_s"] = max([g["duration_s"] for g in gaps], default=0.0)

        # inference rate: rows where the value actually changed
        changed = np.r_[True, (np.abs(np.diff(off)) + np.abs(np.diff(area))) > 0]
        tc = t[changed & v]
        if len(tc) > 3:
            s["vision_update_hz_median"] = round(float(1.0 / np.median(np.diff(tc))), 2)

        # stale coasting: identical consecutive rows while valid = 1
        max_rep = rep = 0
        for i in range(1, len(bbox)):
            if v[i] and v[i - 1] and off[i] == off[i - 1] and area[i] == area[i - 1]:
                rep += 1
                max_rep = max(max_rep, rep)
            else:
                rep = 0
        s["max_repeated_valid_rows"] = max_rep

        if v.any():
            av = area[v]
            s["area_ratio"] = {"min": round(float(av.min()), 2), "p10": round(pct(av, 10), 2),
                               "median": round(float(np.median(av)), 2), "p90": round(pct(av, 90), 2),
                               "max": round(float(av.max()), 2)}
            ov = off[v]
            s["offset_x"] = {"min": round(float(ov.min()), 2), "median": round(float(np.median(ov)), 2),
                             "max": round(float(ov.max()), 2)}

        zone = zone_series(area, valid)
        dts = np.minimum(np.r_[np.diff(t), 0.0], 1.0)
        total = dts.sum() or 1.0
        s["zone_time_share"] = {
            "follow": round(float(dts[zone == 1].sum() / total), 3),
            "hold": round(float(dts[(zone == 0) & v].sum() / total), 3),
            "reverse": round(float(dts[zone == -1].sum() / total), 3),
            "lost": round(float(dts[~v].sum() / total), 3)}

        if s["valid_fraction"] < 0.7:
            flag("warn", "VALID_LOW", f"Target valid only {s['valid_fraction']*100:.0f}% of the run "
                 "(lighting, distance, calibration, or you left the frame).")
        if s["longest_lost_gap_s"] > 3.0:
            flag("warn", "LOST_GAP", f"Longest target-lost stretch {s['longest_lost_gap_s']} s: "
                 + ", ".join(f"{g['start_s']}s+{g['duration_s']}s" + (" (until the run ended)" if g.get("to_end") else "")
                             for g in s["lost_gaps"][:5]))
        if max_rep > 5:
            flag("warn", "STALE_COAST", f"{max_rep} identical valid rows in a row: the tracker is coasting "
                 "on stale data before dropping the target (check LOST_STREAK_LIMIT is loaded).")
        hz = s.get("vision_update_hz_median")
        if hz is not None and hz < 2.0:
            flag("warn", "SLOW_VISION", f"Vision updates at ~{hz} Hz; expect lag.")
    else:
        zone = np.array([])
        flag("error", "NO_BBOX", "No /bbox_error messages recorded.")

    # ---- controller output -------------------------------------------------
    if len(cmd) > 10:
        s["cmd_vel_hz"] = round(len(cmd) / duration, 1)
        s["cmd_lin_x"] = {"p5": round(pct(cmd[:, 1], 5), 3), "median": round(float(np.median(cmd[:, 1])), 3),
                          "p95": round(pct(cmd[:, 1], 95), 3)}
        s["cmd_ang_z_abs_p95"] = round(pct(np.abs(cmd[:, 2]), 95), 3)
        grid, lin_g, ang_g = smooth_grid(cmd)

        if len(bbox) and len(zone):
            lag = 0.5
            lin_at = np.interp(bbox[:, 0] + lag, grid, lin_g)
            ang_at = np.interp(bbox[:, 0] + lag, grid, ang_g)
            v = bbox[:, 4] > 0.5
            fo = (zone == 1) & v
            re_ = (zone == -1) & v
            if fo.sum() > 10:
                frac = float((lin_at[fo] > 0.05).mean())
                s["follow_zone_driving_fraction"] = round(frac, 2)
                if frac < 0.5:
                    flag("warn", "NOT_FOLLOWING", f"In the follow zone the controller drove forward only "
                         f"{frac*100:.0f}% of the time.")
            if re_.sum() > 10:
                frac = float((lin_at[re_] < -0.05).mean())
                s["reverse_zone_reversing_fraction"] = round(frac, 2)
                if frac < 0.5:
                    flag("warn", "NOT_REVERSING", f"In the reverse zone the controller reversed only "
                         f"{frac*100:.0f}% of the time.")
            # turn sign: positive offset_x (person on image right) must give negative angular.z
            sel = v & (np.abs(bbox[:, 1]) > 0.2) & (np.abs(ang_at) > 0.05)
            if sel.sum() > 10:
                agree = float((np.sign(ang_at[sel]) == -np.sign(bbox[sel, 1])).mean())
                s["turn_sign_agreement"] = round(agree, 2)
                if agree < 0.4:
                    flag("error", "TURN_SIGN", f"Turn command points away from the person in "
                         f"{(1-agree)*100:.0f}% of off-center samples: possible left/right flip "
                         "(camera mirror or wheel mapping).")

        # odometry sign: commanded yaw rate vs measured yaw rate, judged only while the
        # rover is really turning (commands below the ESP32 wheel deadband do nothing).
        if len(state) > 10:
            k = max(1, int(0.5 / 0.05))
            r_g = np.convolve(np.interp(grid + 0.2, state[:, 0], state[:, 2]), np.ones(k) / k, "same")
            sel = (np.abs(ang_g) > 0.15) & (np.abs(r_g) > 0.1)
            if sel.sum() >= 20:
                agree = float((np.sign(r_g[sel]) == np.sign(ang_g[sel])).mean())
                s["odom_yaw_sign_agreement"] = round(agree, 2)
                if agree < 0.5:
                    flag("error", "ODOM_SIGN", f"Measured yaw rate disagrees with the commanded turn in "
                         f"{(1-agree)*100:.0f}% of turning samples: odometry left/right may be swapped.")
            # hold-zone turn-assist pulses (lqr_node): short |ang| >= 1.2 rad/s with no forward speed
            p = (np.abs(cmd[:, 2]) >= 1.2) & (np.abs(cmd[:, 1]) < 0.02)
            bursts = runs_of(p)
            if bursts:
                eff, yaws, gains = 0, [], []
                for a, b in bursts:
                    t0, t1 = cmd[a, 0], cmd[b, 0]
                    w = (state[:, 0] >= t0) & (state[:, 0] <= t0 + 1.0)
                    if w.sum() < 3:
                        continue
                    net = float(np.degrees(np.sum(state[w][:-1, 2] * np.diff(state[w, 0]))))
                    yaws.append(net)
                    if np.sign(net) == np.sign(cmd[a, 2]) and abs(net) > 2.0:
                        eff += 1
                    if len(bbox):
                        vb = bbox[:, 4] > 0.5
                        before = bbox[(bbox[:, 0] < t0) & vb]
                        after = bbox[(bbox[:, 0] > t1 + 0.9) & vb]
                        if len(before) and len(after):
                            gains.append(abs(before[-1, 1]) - abs(after[0, 1]))
                n = len(yaws)
                if n:
                    s["turn_pulses"] = {
                        "count": n,
                        "effective_fraction": round(eff / n, 2),
                        "median_net_yaw_deg": round(float(np.median(np.abs(yaws))), 1),
                        "median_offset_improvement": round(float(np.median(gains)), 2) if gains else None}
                    if n >= 3 and eff / n < 0.6:
                        flag("warn", "PULSE_WEAK", f"Only {eff}/{n} turn pulses produced >2 deg of net rotation "
                             "in the commanded direction (post-pulse counter-steering or too weak a pulse).")
            # turn-in-place commands that produced no motion at all (deadband)
            spin = (np.abs(ang_g) > 0.3) & (np.abs(lin_g) < 0.05)
            if spin.sum() >= 30:
                dead = float((np.abs(r_g[spin]) < 0.03).mean())
                s["turn_in_place_ignored_fraction"] = round(dead, 2)
                if dead > 0.5:
                    flag("warn", "TURN_DEADBAND", f"{dead*100:.0f}% of turn-in-place commands (>0.3 rad/s, no "
                         "forward speed) produced no rotation: they are below the wheel-speed deadband "
                         "(about 0.8 rad/s needed). The rover cannot re-center while holding distance.")

    # ---- wheels (ESP32 PID) --------------------------------------------------
    if len(pid) > 10:
        wheels = {}
        stall_total, drive_total = 0, 0
        for name, vi, si, pi in (("A", 1, 3, 5), ("B", 2, 4, 6)):
            vel, sp, pwm = pid[:, vi], pid[:, si], pid[:, pi]
            drive = np.abs(sp) >= SP_DEADBAND
            stall = drive & (np.abs(pwm) >= PWM_SAT) & (np.abs(vel) < 0.02)
            err = np.abs(sp - vel)[drive]
            wheels[name] = {
                "driving_fraction": round(float(drive.mean()), 3),
                "median_abs_error_mps": round(float(np.median(err)), 3) if len(err) else None,
                "pwm_saturated_fraction": round(float((drive & (np.abs(pwm) >= PWM_SAT)).sum() / max(1, drive.sum())), 3),
                "stall_fraction": round(float(stall.sum() / max(1, drive.sum())), 3)}
            stall_total += int(stall.sum())
            drive_total += int(drive.sum())
        s["wheels"] = wheels
        if drive_total and stall_total / drive_total > 0.1:
            flag("error", "MOTOR_STALL", "Wheels at max PWM with ~0 measured speed: motor power/battery, "
                 "wiring, or encoders (check the motor battery first).")
    else:
        flag("info", "NO_PID", "No parsable PID telemetry lines (check /esp32/telemetry was recorded).")

    # ---- warnings ------------------------------------------------------------
    if data["warn"]:
        texts = sorted({w[1] for w in data["warn"]})
        s["esp32_warnings"] = {"count": len(data["warn"]), "unique": texts}
        if any("WATCHDOG" in w for w in texts):
            flag("info", "WATCHDOG", f"ESP32 watchdog stop x{len(data['warn'])} (expected once at the end of a run).")

    s["flags"] = flags
    return s, zone


# --------------------------------------------------------------------------
#  plots
# --------------------------------------------------------------------------
def make_plots(data, zone, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bbox, cmd, state, pid = data["bbox"], data["cmd"], data["state"], data["pid"]
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    fig.suptitle(title)

    if len(bbox):
        t, off, area, valid = bbox[:, 0], bbox[:, 1], bbox[:, 3], bbox[:, 4]
        v = valid > 0.5
        for a, b in runs_of(~v):
            for axx in ax:
                axx.axvspan(t[a], t[min(b + 1, len(t) - 1)], color="red", alpha=0.08, lw=0)
        ax[0].plot(t[v], area[v], ".", ms=3, label="area_ratio")
        for y, c in ((FOLLOW_ENTER, "g"), (FOLLOW_EXIT, "g"), (REVERSE_EXIT, "r"), (REVERSE_ENTER, "r")):
            ax[0].axhline(y, color=c, ls="--", lw=0.7)
        ax[0].set_ylabel("area ratio\n(green=follow, red=reverse)")
        ax[1].plot(t[v], off[v], ".", ms=3, label="offset_x")
    if len(cmd) > 10:
        grid, lin_g, ang_g = smooth_grid(cmd)
        ax[1].plot(grid, ang_g, label="cmd ang.z (smoothed)")
        ax[2].plot(grid, lin_g, label="cmd lin.x (smoothed)")
    if len(state):
        ax[2].plot(state[:, 0], state[:, 1], lw=0.6, alpha=0.6, label="odom v")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_ylabel("offset_x / ang.z")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("linear m/s")
    if len(pid):
        ax[3].plot(pid[:, 0], pid[:, 3], label="spA")
        ax[3].plot(pid[:, 0], pid[:, 1], label="velA")
        ax[4].plot(pid[:, 0], pid[:, 4], label="spB")
        ax[4].plot(pid[:, 0], pid[:, 2], label="velB")
    ax[3].set_ylabel("wheel A m/s")
    ax[4].set_ylabel("wheel B m/s")
    ax[4].set_xlabel("time since start (s)   (red shading = target lost)")
    for axx in ax:
        axx.legend(loc="upper right", fontsize=8)
        axx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------
def print_report(s):
    print(f"\n=== Run summary ({s['duration_s']} s) ===")
    for k in ("valid_fraction", "valid_fraction_whole_run", "initial_wait_s", "vision_update_hz_median",
              "cmd_vel_hz", "max_repeated_valid_rows", "longest_lost_gap_s", "area_ratio", "offset_x",
              "zone_time_share", "follow_zone_driving_fraction",
              "reverse_zone_reversing_fraction", "turn_sign_agreement",
              "odom_yaw_sign_agreement", "turn_in_place_ignored_fraction", "turn_pulses", "wheels"):
        if k in s:
            print(f"  {k}: {s[k]}")
    print("\n--- Flags ---")
    if not s["flags"]:
        print("  none")
    for f in s["flags"]:
        print(f"  [{f['level'].upper()}] {f['code']}: {f['message']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir)
    if not (run / "bag").exists():
        sys.exit(f"No bag/ folder in {run}")

    data, duration = load_run(run)
    summary, zone = analyze(data, duration)
    summary["run"] = run.name
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print_report(summary)
    print(f"\nwrote {run / 'summary.json'}")
    if not args.no_plot:
        make_plots(data, zone, run / "plots.png", run.name)
        print(f"wrote {run / 'plots.png'}")


if __name__ == "__main__":
    main()
