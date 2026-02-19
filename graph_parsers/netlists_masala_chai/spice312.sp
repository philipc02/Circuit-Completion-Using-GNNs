spice
* SPICE Netlist for the Circuit

* Voltage Source
V1 7 3 DC Vs

* Resistors
Rs 7 3 Rs
RL1 5 6 RL1
RL2 2 8 RL2

* NMOS Transistor
M1 3 5 6 6 NMOS

* PMOS Transistor
M2 3 4 2 2 PMOS

* .model statements for NMOS and PMOS
.model NMOS NMOS
.model PMOS PMOS

* Analysis Commands
*.dc
*.ac
*.tran

.end