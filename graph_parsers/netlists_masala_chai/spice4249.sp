spice
* Transistor Q1: npn Q1 collector base emitter
Q1 6 4 3 NPN

* Transistor Q2: npn Q2 collector base emitter
Q2 10 2 3 NPN

* Transistor Q0: npn Q0 collector base emitter
Q0 7 8 5 NPN

* Current Source I_REF
IREF 3 5 DC 1mA

* Resistor R1
R1 3 5 1k

* Voltage Source V_POS
V+ 1 5 DC 10V

* Connect input voltage to node 8 (v_i)
V_IN 8 5 DC 0.7V

* Define model for NPN
.model NPN npn (Is=1e-14 bf=100)