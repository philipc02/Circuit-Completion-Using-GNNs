spice
* SPICE Netlist

* Voltage Source
VDD 4 0 DC 1.8

* Current Source
IREF 4 7 DC IREF

* PMOS Transistor
MREF 4 6 7 7 PMOS

* NMOS Transistors
M1 3 6 2 2 NMOS
M2 5 5 2 2 NMOS

* Analysis Commands
* Add any required analysis here, for example:
*.DC VDD 0 1.8 0.1
*.TRAN 1n 100n
*.END