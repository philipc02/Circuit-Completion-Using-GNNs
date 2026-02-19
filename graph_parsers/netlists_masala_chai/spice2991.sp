spice
* SPICE Netlist
VDD 3 0 DC VDD_val
V1 5 6 DC V1_val
I1 3 6 DC I1_val

* MOSFET
M1 2 5 7 7 NMOS_MODEL

* Resistors
RD1 2 4 RD1_val
RD2 3 2 RD2_val
R1 5 7 R1_val
R2 7 8 R2_val

.model NMOS_MODEL NMOS (level=1)
.END