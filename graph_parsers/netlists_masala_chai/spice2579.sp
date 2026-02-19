spice
*MOSFET Circuit

* NMOS Transistor
M1 4 Vin1 3 3 NMOS_MODEL

* Resistors
RD1 2 4 1k
RS 3 0 1k

* Voltage Source
VDD 2 0 DC 5V

* Input Voltage Source
Vin1 Vin1 0 DC 0V

.model NMOS_MODEL NMOS
.end