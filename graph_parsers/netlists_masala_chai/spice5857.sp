* SPICE Netlist

VDD 3 0 DC 5V
V1 2 0 DC 1V

M1 4 2 6 6 NMOS
M2 3 4 3 3 PMOS

* NMOS model
.model NMOS NMOS (level=1)

* PMOS model
.model PMOS PMOS (level=1)

.end