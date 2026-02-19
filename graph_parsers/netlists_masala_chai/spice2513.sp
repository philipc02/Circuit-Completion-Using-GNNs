spice
* Current Source
I1 3 7 DC 1A

* MOSFETs
M1 2 6 1 1 NMOS
M2 3 5 2 2 NMOS

* Resistors
r_o1 2 1 1k
r_o2 3 2 1k

* Voltage Sources (for simulation purposes)
VDD 7 0 DC 5V
VIN 6 0 DC 1V
VB 5 0 DC 1V

* Output
Iout 4 0

* .model statement for NMOS
.model NMOS NMOS (KP=1 VTO=0.7)

.end