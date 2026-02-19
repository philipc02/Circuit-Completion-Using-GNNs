plaintext
* NMOS Transistor
M1 2 6 5 5 NMOS

* PMOS Transistor
M2 3 2 4 4 PMOS

* Current Sources
I1 0 2 DC 2I
I2 4 0 DC I

* Voltage Sources
Vvi 6 0 DC vi
Vbias 3 0 DC Vbias

* Model Definitions
.model NMOS NMOS
.model PMOS PMOS

.end