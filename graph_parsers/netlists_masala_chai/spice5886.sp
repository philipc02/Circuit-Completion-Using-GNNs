spice
* NMOS Transistor
M1 2 4 6 6 NMOS

* Voltage Source Vsig
Vsig 3 0 DC 0

* Resistors
R1 4 0 0.5MEG
R2 4 2 0.5MEG
RD 1 2 RD_value

* Voltage Source VDD
VDD 1 2 DC 10

* .Model statement for NMOS
.model NMOS NMOS (Level=1) 

* .End statement
.end