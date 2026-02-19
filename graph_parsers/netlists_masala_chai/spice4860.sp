spice
* Simple BJT Amplifier with Zener Diode
* Transistors
Q1 3 4 2 NPN
Q2 5 8 3 NPN

* Resistors
R1 2 10 1k
R2 10 3 1k
R3 8 7 1k
R4 3 6 1k
RL 7 5 1k

* Diode
D1 3 9 D_z

* Voltage Sources
V_in 1 2 DC 5
V_out 5 7 DC

* Zener Diode Model
.model D_z D(BV=5)

.end