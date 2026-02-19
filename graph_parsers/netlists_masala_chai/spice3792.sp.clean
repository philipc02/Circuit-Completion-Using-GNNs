spice
* Netlist for the given circuit

* Voltage Source
V1 5 0 DC vi

* Current Source
I1 2 4 DC IQ

* Resistors
RG 7 0 200k
RD 6 3 RD_value
RL 3 4 RL_value

* Capacitors
CC1 5 7 CC1_value
CC2 3 4 CC2_value
CS 4 0 CS_value

* NMOS Transistor
M1 3 7 2 2 NMOS_model

* Define power supply nodes
V+ 6 0 DC V_plus
V- 0 4 DC V_minus

* Model definitions - replace `NMOS_model` with specific model details
.model NMOS_model NMOS (kp=1u vto=1 threshold parameters...)

.end