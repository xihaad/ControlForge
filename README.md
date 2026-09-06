# ControlForge

## Proportional Controller

A simulation of three rooms heated by a proportional (P) controller, modeling
each room's thermal dynamics (`C * dT/dt = Q_heater - k*(T_room - T_outside)`)
and animating temperature, setpoint, and heater power over a 1-hour window.

Room 1 and Room 2 share the same setpoint schedule but use a weak vs. strong
gain (`Kp`) to show how gain affects response speed. Room 3 uses a strong
gain paired with an undersized heater and a simulated cold front, showing
actuator saturation and the steady-state droop inherent to pure P control.

### Running

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python proportional_controller\room_heating.py
```