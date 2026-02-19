spice
* NMOS Transistor
M1 5 6 7 7 NMOS

* Resistors
R1 22 2 1k
RG Vin0 3 1k
RD 22 5 1k
R2 3 0 1k
RS 7 0 1k

* Capacitor
C1 3 6 1u

* Voltage Source
VDD 22 0 DC 5V

* Simulation commands
.tran 1n 100n
.end