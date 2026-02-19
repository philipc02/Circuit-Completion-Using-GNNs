spice
* Example SPICE Netlist
VDD 5 0 DC 10
VEE 6 0 DC -10

RG 2 0 1k
RD 5 3 1k
RE 4 6 1k

M1 3 2 4 4 NMOS
M2 4 3 5 5 PMOS

.model NMOS NMOS
.model PMOS PMOS