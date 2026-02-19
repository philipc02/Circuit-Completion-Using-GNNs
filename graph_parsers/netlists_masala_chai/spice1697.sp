plaintext
* SPICE Netlist

VDD 1 0 DC 5V
VIN 3 0 AC 1

M1 Y X 0 0 NMOS
M2 2 Y 1 1 PMOS

RS 3 X RS_VALUE
RL 1 2 RL_VALUE

* NMOS and PMOS model definitions
.model NMOS NMOS (LEVEL=1)
.model PMOS PMOS (LEVEL=1)

.end