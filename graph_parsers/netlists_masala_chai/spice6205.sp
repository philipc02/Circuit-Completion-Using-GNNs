spice
* Nodes are identified by their red annotations in the helper image

* Voltage Source
Vs 6 9 DC <voltage_value>

* Current Source
I1 3 0 DC <current_value>

* Supply Voltage
VDD 4 0 DC <voltage_value>

* NMOS Transistor
* (Drain, Gate, Source)
M1 3 2 0 0 NMOS_Model

* PMOS Transistor
* (Drain, Gate, Source)
M2 5 3 4 4 PMOS_Model

* Capacitor
CC 2 3 <capacitance_value>

* Resistors
R1 3 7 <resistance_value>
R2 2 3 <resistance_value>
RL 2 0 <resistance_value>
Rin 6 2 <resistance_value>

* Model Definitions
.model NMOS_Model NMOS (LEVEL=1)
.model PMOS_Model PMOS (LEVEL=1)

* .end