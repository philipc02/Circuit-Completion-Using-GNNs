spice
* SPICE Netlist

.model PMOS PMOS
.model NMOS NMOS

VREF 6 0 DC V+
IREF 6 0 DC IREF
R1 6 2 R
M1 2 2 6 6 PMOS
M2 4 2 3 3 NMOS
Rout 4 0 Ro
Iout 4 0 IO

* End of Netlist