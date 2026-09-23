import numpy as np

# ============================================================
#  ROOM HEATING WITH PROPORTIONAL (P) CONTROL — simulation only
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
#  This module holds only the numerical simulation, so it can be
#  reused by different front ends (matplotlib animation, MuJoCo
#  visualization, ...) without duplicating the control algorithm.
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
