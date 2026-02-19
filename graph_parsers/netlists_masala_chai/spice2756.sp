spice
* SPICE netlist for the schematic

VIN 5 0 DC 0
VDD 3 0 DC 1.8

RS 5 4 1k
RD 3 6 1k

M1 6 2 0 NMOS

C1 4 0 1p
C2 6 0 1p

* Nodes:
* 0 - Ground
* 2 - Gate of M1