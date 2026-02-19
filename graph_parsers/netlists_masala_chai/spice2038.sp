spice
* NMOS Circuit
M1 2 1 3 3 NMOS

RD 4 2 1k

CL 2 0 10pF

VDD 4 0 DC 5V

* Input Voltage Source
Vin 1 0 DC 1V

* .MODEL statements for NMOS
.model NMOS NMOS (LEVEL=1)