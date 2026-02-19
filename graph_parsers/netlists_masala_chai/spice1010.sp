spice
* Example SPICE netlist
VCC 3 0 DC 2.5
M1 3 3 4 4 NMOS
M2 4 2 5 5 NMOS
R1 5 0 300
R2 2 0 80k
.model NMOS NMOS (level=1)

* End of netlist