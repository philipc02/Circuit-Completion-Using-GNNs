spice
* NMOS Transistor
M1 7 6 4 4 NMOS

* BJT Transistor
Q1 5 6 4 NPN

* Voltage Sources
Vdd 8 4 DC 5V
Vi 7 4 DC <Vi>

* Resistors
R1 6 4 300
R2 6 4 100
R3 8 5 2k

* Output
Vo 5 4

.model NMOS NMOS(Level=1)
.model NPN NPN

.end