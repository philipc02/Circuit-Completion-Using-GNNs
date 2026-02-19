* NMOS Circuit
M1 2 3 4 4 NMOS

* Resistors
RD 2 VDD 1k
RS 4 0 1k

* Voltage source
VDD VDD 0 DC 5V
Vin 3 0 DC 0V AC 1V

* Models
.model NMOS nmos level=1

.end