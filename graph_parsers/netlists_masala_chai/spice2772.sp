spice
* NMOS Transistor
M1 3 2 5 5 NMOS

* Input Voltage Source
Vin 6 7 DC 0

* Resistors
RD VDD 3 1k
RS 6 4 1k

* Capacitors
CD 3 1 1u
CS 5 7 1u

* Voltage source for VDD
VDD VDD 0 DC 5

* Node Definitions
Vout 1 0
Vb 2 0

* Model Definitions (assumed for NMOS)
.model NMOS NMOS (LEVEL=1 TOX=90n KP=120u W=1u L=1u)