spice
* SPICE netlist for the given circuit
VDD 2 0 DC 5V
Vin 1 0 DC 0V

M1 2 1 0 0 NMOS

RD 2 3 1k
CL 3 0 10p

.model NMOS NMOS