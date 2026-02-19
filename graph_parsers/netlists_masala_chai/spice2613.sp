* SPICE netlist for the given circuit

VDD 3 0 DC 5V

RD1 3 4 10k
RD2 6 4 10k

M1 4 1 4 4 NMOS
M2 4 5 6 6 NMOS
M3 4 2 0 0 NMOS

Vin 1 0 DC 0V
Vcont 2 0 DC 0V

.model NMOS NMOS level=1

.end