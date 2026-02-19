spice
* PMOS Transistors
M_L1 2 5 1 1 PMOS
M_L2 2 4 6 6 PMOS

* NMOS Transistors
M_D1 3 5 3 3 NMOS
M_D2 3 4 7 7 NMOS

* Voltage Source
V_DD 2 0 DC 5

* Input Voltage
V_I 5 0 DC <value>

* Output Nodes
V_O1 3
V_O2 4

* Model Definitions
.model PMOS PMOS
.model NMOS NMOS