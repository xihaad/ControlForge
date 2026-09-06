import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np

# ============================================================
#  ROOM HEATING WITH PROPORTIONAL (P) CONTROL
#
#  Physics of one room:
#      C * dT/dt = Q_heater - k*(T_room - T_outside)
#
#      C = thermal capacitance [J/K]  (air + walls + furniture)
#      k = heat loss coefficient [W/K] (through walls/windows)
#      Q = heater power [W], the control input
#
#  Controller:  Q = Kp * (T_ref - T_room),  clipped to [0, Q_max]
#
#  Two things happen here that did NOT happen in the water tanks:
#    1. The room LEAKS heat, so it is not a pure integrator.
#       -> a P controller leaves permanent steady-state error (droop).
#    2. A heater cannot cool, and has a maximum power.
#       -> the response is asymmetric and can saturate.
# ============================================================

# ---------- Simulation timing ----------
dt = 2.0                              # Time step [s]
t0 = 0.0                              # Start time [s]
t_end = 3600.0                        # End time [s]  (1 hour)
t = np.arange(t0, t_end + dt, dt)     # Time vector, 1801 elements
frame_amount = len(t) - 1             # Animation frames

# ---------- Building physics (shared by all rooms) ----------
C = 3.0e5                             # Thermal capacitance [J/K]
k = 250.0                             # Heat loss coefficient [W/K]
T_out_base = 2.0                      # Baseline outside temperature [C]

# Open-loop time constant of an unheated room:
#     tau_open = C/k = 1200 s = 20 min
# Closed-loop time constant with a P controller:
#     tau_cl = C/(Kp + k)   <-- higher gain = faster room

# ---------- Controller gains and heater sizes ----------
Kp1 = 1500.0                          # Room 1: weak gain      -> tau = 171 s
Kp2 = 6000.0                          # Room 2: strong gain    -> tau = 48 s
Kp3 = 6000.0                          # Room 3: strong gain, but weak heater

Q_max1 = 9000.0                       # Heater power limit, room 1 [W]
Q_max2 = 9000.0                       # Heater power limit, room 2 [W]
Q_max3 = 6000.0                       # Undersized heater, room 3 [W]

# ---------- Initial conditions ----------
T1_i = 12.0                           # Room 1 starts cold
T2_i = 12.0                           # Room 2 starts cold (same, for fair comparison)
T3_i = 22.0                           # Room 3 starts already at its setpoint

# ---------- Outside temperature ----------
# Rooms 1 and 2: steady cold day.
T_out12 = np.full(len(t), T_out_base)

# Room 3: a cold front sweeps through, bottoming out at -8 C near t = 1800 s.
T_out3 = T_out_base - 10.0 * np.exp(-((t - 1800.0) / 450.0) ** 2)

# ---------- Setpoint for room 3 (constant) ----------
T_ref3 = np.full(len(t), 22.0)

# ---------- Storage vectors ----------
# Room 1
T_ref1 = np.zeros(len(t)); T_ref1[0] = 21.0
T_room1 = np.zeros(len(t)); T_room1[0] = T1_i
error1 = np.zeros(len(t))
Q1 = np.zeros(len(t))

# Room 2
T_ref2 = np.zeros(len(t)); T_ref2[0] = 21.0
T_room2 = np.zeros(len(t)); T_room2[0] = T2_i
error2 = np.zeros(len(t))
Q2 = np.zeros(len(t))

# Room 3
T_room3 = np.zeros(len(t)); T_room3[0] = T3_i
error3 = np.zeros(len(t))
Q3 = np.zeros(len(t))

# Seed the first control action so the first integration step is correct.
Q1[0] = np.clip(Kp1 * (T_ref1[0] - T_room1[0]), 0.0, Q_max1)
Q2[0] = np.clip(Kp2 * (T_ref2[0] - T_room2[0]), 0.0, Q_max2)
Q3[0] = np.clip(Kp3 * (T_ref3[0] - T_room3[0]), 0.0, Q_max3)


def step_room(T_prev, Q_prev, Q_now, To_prev, To_now):
    """Advance one room by dt using Heun's method (predictor-corrector).

    The water tank code could use a plain trapezoid because the derivative
    depended only on the control input. Here dT/dt depends on T itself
    (the heat-loss term), so a true trapezoid would be implicit. Heun
    fixes this: take an Euler guess, then average the two slopes.
    """
    f1 = (Q_prev - k * (T_prev - To_prev)) / C          # slope at the start
    T_pred = T_prev + f1 * dt                            # Euler predictor
    f2 = (Q_now - k * (T_pred - To_now)) / C             # slope at the end
    return T_prev + 0.5 * (f1 + f2) * dt                 # trapezoid corrector


# ============================================================
#  SIMULATION LOOP
# ============================================================
for i in range(1, len(t)):

    # ---- Thermostat schedule for rooms 1 and 2 (identical on purpose) ----
    # Both rooms follow the same setpoint staircase, so the ONLY difference
    # between them is Kp. That makes the comparison a controlled experiment.
    if t[i] < 900.0:
        T_ref1[i] = 21.0        # Morning: warm up
    elif t[i] < 1800.0:
        T_ref1[i] = 17.0        # Setback: nobody home
    elif t[i] < 2700.0:
        T_ref1[i] = 23.0        # Evening: comfort boost
    else:
        T_ref1[i] = 19.0        # Night: cool down
    T_ref2[i] = T_ref1[i]

    # ---- Error: where we want to be minus where we are ----
    error1[i - 1] = T_ref1[i - 1] - T_room1[i - 1]
    error2[i - 1] = T_ref2[i - 1] - T_room2[i - 1]
    error3[i - 1] = T_ref3[i - 1] - T_room3[i - 1]

    # ---- P control law, with actuator limits ----
    # np.clip is the whole story of "a heater cannot cool and cannot
    # produce infinite power". Lower limit 0, upper limit Q_max.
    Q1[i] = np.clip(Kp1 * error1[i - 1], 0.0, Q_max1)
    Q2[i] = np.clip(Kp2 * error2[i - 1], 0.0, Q_max2)
    Q3[i] = np.clip(Kp3 * error3[i - 1], 0.0, Q_max3)

    # ---- Integrate the heat balance forward one step ----
    T_room1[i] = step_room(T_room1[i - 1], Q1[i - 1], Q1[i], T_out12[i - 1], T_out12[i])
    T_room2[i] = step_room(T_room2[i - 1], Q2[i - 1], Q2[i], T_out12[i - 1], T_out12[i])
    T_room3[i] = step_room(T_room3[i - 1], Q3[i - 1], Q3[i], T_out3[i - 1], T_out3[i])

# Fill the final error entries so the arrays are complete.
error1[-1] = T_ref1[-1] - T_room1[-1]
error2[-1] = T_ref2[-1] - T_room2[-1]
error3[-1] = T_ref3[-1] - T_room3[-1]

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