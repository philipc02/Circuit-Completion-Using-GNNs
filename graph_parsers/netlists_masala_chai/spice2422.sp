* SPICE Netlist for the Given Schematic

V1 N1 0 DC 1V
V2 N3 0 DC 1.9V
M1 N1 N1 N3 N3 NMOS
Vx N2 0 DC 0V
Ix N1 N2 DC 0A

* NMOS model
.model NMOS NMOS (level=1)

.end