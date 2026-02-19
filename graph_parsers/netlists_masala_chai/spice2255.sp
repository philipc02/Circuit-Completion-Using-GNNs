spice
* SPICE Netlist for the Circuit

VDD 3 0 DC 5

M1 3 2 2 2 NMOS
M2 2 4 0 0 NMOS

R1 4 2 1k
R2 4 0 1k

Vin 2 0 DC 0

.model NMOS NMOS

.end