spice
* Netlist for the given circuit
V1 2 0 DC 5
V2 0 5 DC 5
IREF 2 0 DC 1

M1 6 8 2 2 PMOS
M2 7 8 6 6 PMOS
M3 4 3 5 5 NMOS
M4 2 3 4 4 NMOS

RD 6 2 1k
RG 2 0 1k

.END