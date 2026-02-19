plaintext
*MOSFET Differential Amplifier

* Voltage Source
VDD 2 0 DC 5V

* NMOS Transistors
M1 5 1 3 3 NMOS
M2 3 4 3 3 NMOS

* Resistors
RD1 2 5 1k
RS1 3 0 1k
RF1 3 3 1k
RF2 5 4 1k
RD2 4 6 1k

* Voltage Output
Vout 3 0

* Input Voltage
Vin 1 0 DC 1V

* Model Parameters
.model NMOS NMOS

.end