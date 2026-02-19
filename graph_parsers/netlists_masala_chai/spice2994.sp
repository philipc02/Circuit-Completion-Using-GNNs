plaintext
* CMOS Amplifier Netlist

* NMOS transistor
M1 2 5 0 0 NMOS

* PMOS transistor
M2 3 2 4 4 PMOS

* Resistors
RD 4 2 1k
R1 6 5 1k
R2 2 3 1k
RS 3 0 1k

* Voltage Source
Vin 6 0 DC 0V

* Power Supply
VDD 4 0 DC 5V

.model NMOS NMOS
.model PMOS PMOS

.end