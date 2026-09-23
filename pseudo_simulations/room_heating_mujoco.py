"""3D visualization of the proportional room-heating controller in MuJoCo.

MuJoCo is a rigid-body physics engine, not a thermal simulator, so this
script does NOT use MuJoCo's dynamics (mj_step). The control algorithm and
room physics are the exact same code as proportional_controller/room_heating.py
(imported from room_heating_sim.py, its shared simulation module). MuJoCo is
used purely as a 3D renderer: each simulated frame, the precomputed state is
written directly into geom sizes/positions/colors (no joints or actuators
involved) and the scene is redrawn.

Per room you will see:
  - a rising column ("thermometer") for room temperature, colored on the
    same blue->red scale as the matplotlib version
  - a red tab riding the column at the current setpoint
  - a grey->orange bar showing heater power, with a black tab marking that
    room's power ceiling (Q_max) so saturation is visible when the bar
    hits the cap
  - (room 3 only) a blue tab showing the outside temperature, so the cold
    front sweeping through is visible as it approaches the room's column
"""
import os
import sys
import time

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "proportional_controller"))
import room_heating_sim as sim  # noqa: E402  (path must be set up first)

SCENE_PATH = os.path.join(os.path.dirname(__file__), "room_heating_scene.xml")

# ---------- Same axis ranges as the matplotlib thermometers ----------
T_MIN, T_MAX = -10.0, 30.0
Q_MIN, Q_MAX = 0.0, 10000.0
H_MAX = 2.0                      # column height in the scene, in meters
MIN_H = 0.015                    # avoid zero-size geoms

cmap = cm.coolwarm
norm = mcolors.Normalize(10.0, 26.0)
heater_off = np.array([0.5, 0.5, 0.5])
heater_on = np.array([1.0, 0.45, 0.0])


def temp_to_h(T):
    frac = np.clip((T - T_MIN) / (T_MAX - T_MIN), 0.0, 1.0)
    return max(frac * H_MAX, MIN_H)


def q_to_h(Q):
    frac = np.clip((Q - Q_MIN) / (Q_MAX - Q_MIN), 0.0, 1.0)
    return max(frac * H_MAX, MIN_H)


def set_column(m, prefix, h, rgba=None):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, prefix)
    m.geom_size[gid, 2] = h / 2.0
    m.geom_pos[gid, 2] = h / 2.0
    if rgba is not None:
        m.geom_rgba[gid, :3] = rgba


def set_marker(m, name, h):
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    m.geom_pos[gid, 2] = h


def main():
    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)

    n = len(sim.t)
    print("Playing back %d simulated steps (%.0f s of simulated time) ..." % (n, sim.t_end))

    with mujoco.viewer.launch_passive(m, d) as v:
        v.opt.label = mujoco.mjtLabel.mjLABEL_BODY

        i = 0
        last_report = time.time()
        while v.is_running():
            # ---- Room 1 ----
            set_column(m, "r1_fill", temp_to_h(sim.T_room1[i]), cmap(norm(sim.T_room1[i]))[:3])
            set_marker(m, "r1_setpoint", temp_to_h(sim.T_ref1[i]))
            frac1 = np.clip(sim.Q1[i] / sim.Q_max1, 0.0, 1.0)
            set_column(m, "r1_heater_fill", q_to_h(sim.Q1[i]), heater_off + frac1 * (heater_on - heater_off))

            # ---- Room 2 ----
            set_column(m, "r2_fill", temp_to_h(sim.T_room2[i]), cmap(norm(sim.T_room2[i]))[:3])
            set_marker(m, "r2_setpoint", temp_to_h(sim.T_ref2[i]))
            frac2 = np.clip(sim.Q2[i] / sim.Q_max2, 0.0, 1.0)
            set_column(m, "r2_heater_fill", q_to_h(sim.Q2[i]), heater_off + frac2 * (heater_on - heater_off))

            # ---- Room 3 ----
            set_column(m, "r3_fill", temp_to_h(sim.T_room3[i]), cmap(norm(sim.T_room3[i]))[:3])
            set_marker(m, "r3_setpoint", temp_to_h(sim.T_ref3[i]))
            set_marker(m, "r3_outside", temp_to_h(sim.T_out3[i]))
            frac3 = np.clip(sim.Q3[i] / sim.Q_max3, 0.0, 1.0)
            set_column(m, "r3_heater_fill", q_to_h(sim.Q3[i]), heater_off + frac3 * (heater_on - heater_off))

            mujoco.mj_forward(m, d)
            v.sync()

            if time.time() - last_report > 1.0:
                print(
                    "t=%5.0fs | R1 %5.1fC (set %.1f, %.1fkW)  "
                    "R2 %5.1fC (set %.1f, %.1fkW)  "
                    "R3 %5.1fC (set %.1f, %.1fkW, out %.1fC)"
                    % (
                        sim.t[i],
                        sim.T_room1[i], sim.T_ref1[i], sim.Q1[i] / 1000.0,
                        sim.T_room2[i], sim.T_ref2[i], sim.Q2[i] / 1000.0,
                        sim.T_room3[i], sim.T_ref3[i], sim.Q3[i] / 1000.0, sim.T_out3[i],
                    )
                )
                last_report = time.time()

            i += 1
            if i >= n:
                i = 0  # loop the playback, like the matplotlib animation

            time.sleep(0.02)  # ~36s to play back the full simulated hour


if __name__ == "__main__":
    main()
