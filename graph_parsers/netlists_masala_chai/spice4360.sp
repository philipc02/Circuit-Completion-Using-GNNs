spice
* SPICE netlist
M1 5 9 7 7 NMOS
M2 6 2 7 7 NMOS
RD1 8 5 RD
RD2 8 6 RD
IQ 7 1 DC
VDD 8 0 DC V+
VSS 1 0 DC V-
VIN1 9 0 DC vd/2
VIN2 2 0 DC vd2

* NMOS model
.model NMOS NMOS(Level=1 Vto=0.7)

.END