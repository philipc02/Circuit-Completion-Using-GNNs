spice
* NMOS Transistors
M1 4 2 3 3 NMOS
M2 8 6 3 3 NMOS

* Resistors
R1 3 5 35k
R2 3 6 5k

* Current Sources
I1 5 3 DC
I2 8 3 DC

* Voltage Sources
V+ 9 4 DC 5
V- 5 3 DC -5
Vin 2 3 DC 0

* Model Definitions
.model NMOS NMOS
.end