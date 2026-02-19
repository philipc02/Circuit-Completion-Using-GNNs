spice
* SPICE Netlist
VDD 4 0 DC VDD

* Resistors
R1 4 2 RD
R2 4 22 RD
R3 4 3 RD

* MOSFETs
M1 2 1 0 0 NMOS
M2 22 2 0 0 NMOS
M3 3 22 0 0 NMOS

* Capacitors
C1 2 0 C1
C2 22 0 C1
C3 3 0 C1

* Voltage and Current Sources
Vin 1 0 DC Vin

.END