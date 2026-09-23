import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np

from room_heating_sim import (
    t, t0, t_end, frame_amount,
    T_out_base, T_out12, T_out3,
    Kp1, Kp2, Kp3, Q_max1, Q_max2, Q_max3,
    T_ref1, T_ref2, T_ref3,
    T_room1, T_room2, T_room3,
    Q1, Q2, Q3,
)

# ============================================================
#  ROOM HEATING WITH PROPORTIONAL (P) CONTROL — visualization
#
#  The simulation itself (physics + control law) lives in
#  room_heating_sim.py, so it can be reused by other front ends
#  (e.g. the MuJoCo visualization in pseudo_simulations/).
#
#  Two things happen here that did NOT happen in the water tanks:
#    1. The room LEAKS heat, so it is not a pure integrator.
#       -> a P controller leaves permanent steady-state error (droop).
#    2. A heater cannot cool, and has a maximum power.
#       -> the response is asymmetric and can saturate.
# ============================================================

print('Steps simulated:', len(t))
print('Room 1 final: %.2f C (setpoint %.1f)' % (T_room1[-1], T_ref1[-1]))
print('Room 2 final: %.2f C (setpoint %.1f)' % (T_room2[-1], T_ref2[-1]))
print('Room 3 final: %.2f C (setpoint %.1f)' % (T_room3[-1], T_ref3[-1]))


# ============================================================
#  FIGURE SETUP
# ============================================================
T_min, T_max_plot = -10.0, 30.0       # y-limits for the thermometers
cmap = plt.cm.coolwarm                # Cold = blue, warm = red
norm = plt.Normalize(10.0, 26.0)      # Maps room temperature to a colour

fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=(0.85, 0.85, 0.85))
gs = gridspec.GridSpec(3, 3, height_ratios=[1.5, 1, 1], hspace=0.45, wspace=0.25)


def make_thermometer(cell, title):
    """Build one thermometer panel and return its animated artists."""
    ax = fig.add_subplot(cell, facecolor=(0.94, 0.94, 0.94))
    # solid_capstyle='butt' removes the projecting line cap, so the top of
    # the bar sits exactly at the temperature value. (The tank script had to
    # subtract a magic 63 to work around the default projecting cap.)
    bar, = ax.plot([], [], linewidth=170, zorder=0,
                   solid_capstyle='butt', color='tomato')
    ref, = ax.plot([], [], 'r--', linewidth=2, zorder=3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(T_min, T_max_plot)
    ax.set_xticks([])
    ax.set_yticks(np.arange(T_min, T_max_plot + 5, 5))
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_title(title, fontsize=11)
    txt = ax.text(-0.95, 26.5, '', fontsize=10, family='monospace', zorder=4)
    return ax, bar, ref, txt


# ---- Room 1 thermometer ----
ax0, bar1, ref1, txt1 = make_thermometer(gs[0, 0], 'Room 1  |  Kp = 1500 (weak gain)')
ax0.set_ylabel('temperature [$^\\circ$C]')
ax0.axhline(T_out_base, color='deepskyblue', linestyle=':', linewidth=1.5)

# ---- Room 2 thermometer ----
ax1, bar2, ref2, txt2 = make_thermometer(gs[0, 1], 'Room 2  |  Kp = 6000 (strong gain)')
ax1.axhline(T_out_base, color='deepskyblue', linestyle=':', linewidth=1.5)

# ---- Room 3 thermometer ----
ax2, bar3, ref3, txt3 = make_thermometer(gs[0, 2], 'Room 3  |  Kp = 6000, 6 kW heater + cold front')
out3_marker, = ax2.plot([], [], color='deepskyblue', linestyle=':', linewidth=2)

# ---- Temperature history (spans the middle row) ----
ax3 = fig.add_subplot(gs[1, :], facecolor=(0.94, 0.94, 0.94))
ref_hist, = ax3.plot([], [], 'k--', linewidth=2, label='setpoint (rooms 1 & 2)')
T1_hist, = ax3.plot([], [], 'blue', linewidth=3, label='Room 1')
T2_hist, = ax3.plot([], [], 'green', linewidth=3, label='Room 2')
T3_hist, = ax3.plot([], [], 'red', linewidth=3, label='Room 3')
out_hist, = ax3.plot([], [], color='deepskyblue', linestyle=':', linewidth=2,
                     label='outside (room 3)')
ax3.axhline(22.0, color='grey', linestyle='--', linewidth=1.5)
ax3.set_xlim(t0, t_end)
ax3.set_ylim(T_min, T_max_plot)
ax3.set_ylabel('temperature [$^\\circ$C]')
ax3.grid(True, alpha=0.4)
ax3.legend(loc='lower right', fontsize='small', ncol=5)

# ---- Heater power history (spans the bottom row) ----
ax4 = fig.add_subplot(gs[2, :], facecolor=(0.94, 0.94, 0.94))
Q1_hist, = ax4.plot([], [], 'blue', linewidth=3, label='Room 1')
Q2_hist, = ax4.plot([], [], 'green', linewidth=3, label='Room 2')
Q3_hist, = ax4.plot([], [], 'red', linewidth=3, label='Room 3')
ax4.axhline(Q_max1 / 1000.0, color='black', linestyle='--', linewidth=1.5)
ax4.axhline(Q_max3 / 1000.0, color='red', linestyle='--', linewidth=1.0, alpha=0.6)
ax4.text(60, Q_max1 / 1000.0 + 0.2, 'heater limit, rooms 1 & 2', fontsize=8)
ax4.text(60, Q_max3 / 1000.0 + 0.2, 'heater limit, room 3', fontsize=8, color='red')
ax4.set_xlim(t0, t_end)
ax4.set_ylim(0, 10.5)
ax4.set_xlabel('time [s]')
ax4.set_ylabel('heater power [kW]')
ax4.grid(True, alpha=0.4)
ax4.legend(loc='upper right', fontsize='small', ncol=3)

fig.text(0.01, 0.97, 'Room heating with proportional control', size=13, weight='bold')


# ============================================================
#  ANIMATION
# ============================================================
def update_plot(num):
    if num >= len(t):
        num = len(t) - 1

    # --- Room 1 thermometer ---
    bar1.set_data([0, 0], [T_min, T_room1[num]])
    bar1.set_color(cmap(norm(T_room1[num])))
    ref1.set_data([-1, 1], [T_ref1[num], T_ref1[num]])
    txt1.set_text('T =%6.1f C\nset%6.1f C\nQ =%6.1f kW'
                  % (T_room1[num], T_ref1[num], Q1[num] / 1000.0))

    # --- Room 2 thermometer ---
    bar2.set_data([0, 0], [T_min, T_room2[num]])
    bar2.set_color(cmap(norm(T_room2[num])))
    ref2.set_data([-1, 1], [T_ref2[num], T_ref2[num]])
    txt2.set_text('T =%6.1f C\nset%6.1f C\nQ =%6.1f kW'
                  % (T_room2[num], T_ref2[num], Q2[num] / 1000.0))

    # --- Room 3 thermometer ---
    bar3.set_data([0, 0], [T_min, T_room3[num]])
    bar3.set_color(cmap(norm(T_room3[num])))
    ref3.set_data([-1, 1], [T_ref3[num], T_ref3[num]])
    out3_marker.set_data([-1, 1], [T_out3[num], T_out3[num]])
    txt3.set_text('T =%6.1f C\nset%6.1f C\nQ =%6.1f kW\nout%6.1f C'
                  % (T_room3[num], T_ref3[num], Q3[num] / 1000.0, T_out3[num]))

    # --- Temperature history ---
    ref_hist.set_data(t[0:num], T_ref1[0:num])
    T1_hist.set_data(t[0:num], T_room1[0:num])
    T2_hist.set_data(t[0:num], T_room2[0:num])
    T3_hist.set_data(t[0:num], T_room3[0:num])
    out_hist.set_data(t[0:num], T_out3[0:num])

    # --- Heater power history ---
    Q1_hist.set_data(t[0:num], Q1[0:num] / 1000.0)
    Q2_hist.set_data(t[0:num], Q2[0:num] / 1000.0)
    Q3_hist.set_data(t[0:num], Q3[0:num] / 1000.0)

    return (bar1, ref1, txt1,
            bar2, ref2, txt2,
            bar3, ref3, out3_marker, txt3,
            ref_hist, T1_hist, T2_hist, T3_hist, out_hist,
            Q1_hist, Q2_hist, Q3_hist)


room_ani = animation.FuncAnimation(fig, update_plot,
                                   frames=frame_amount, interval=20,
                                   repeat=True, blit=True)
plt.show()