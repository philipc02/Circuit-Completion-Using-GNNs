spice
* SPICE netlist

V1 Vin 1 0
C1 1 4 1pF
R1 4 5 1k
RF 5 3

M1 2 5 0 0 NMOS
M2 3 3 23 23 PMOS
M3 6 2 0 0 NMOS
M4 4 3 23 23 PMOS

VDD 23 0 DC 1.8

* End of netlist