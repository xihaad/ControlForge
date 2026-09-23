import time, numpy as np, mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_path("pseudo_simulations/shadow_hand/my_scene.xml")
d = mujoco.MjData(m)
lo, hi = m.actuator_ctrlrange.T
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        d.ctrl[:] = lo + (hi - lo) * (0.5 + 0.5 * np.sin(d.time))
        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(m.opt.timestep)