*SPICE Netlist
Vx 2 7 DC 0
R10 5 3 50k

*Assuming Q18 is a PMOS
MP18 4 5 2 2 PMOSMODEL

*Assuming Q19 is an NMOS
MN19 5 6 2 2 NMOSMODEL

* Model declarations (to be provided)
.model PMOSMODEL PMOS
.model NMOSMODEL NMOS

.END