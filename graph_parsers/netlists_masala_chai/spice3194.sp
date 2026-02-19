spice
* SPICE Netlist

M3 2 X 0 0 NMOS
Me 1 3 X X NMOS
Mc 4 3 2 2 PMOS
Ma 4 4 4 4 PMOS
Q1 X 5 6 NPN
R3 X 5 1k
Iss 4 0 DC 10mA

.model NMOS NMOS
.model PMOS PMOS
.model NPN NPN

.end