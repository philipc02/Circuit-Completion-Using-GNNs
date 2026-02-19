plaintext
* Example SPICE netlist
VDD 3 0 DC 5V
Vin1 4 0 DC 0V
Vin2 2 0 DC 0V

RD1 5 3 1k
RD2 6 3 1k

M1 5 4 8 8 NMOS
M2 6 2 7 7 NMOS

.model NMOS NMOS LEVEL=1

.END