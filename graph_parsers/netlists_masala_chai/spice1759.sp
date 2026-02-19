spice
* NMOS Transistors
M1 5 4 0 0 NMOS
M2 6 3 0 0 NMOS

* Resistors
R1 Vin 4 5k
R2 5 2 1k
R3 6 2 1k

* Voltage Source
V1 2 0 DC 2.5

* Output
.control
probe V(3)
.endc

* NMOS Model
.model NMOS NMOS (LEVEL=1)
.end