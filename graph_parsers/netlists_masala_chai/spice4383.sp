spice
* SPICE Netlist for the Schematic

* Voltage Sources
V1 3 4 DC 2.5
Vi 3 0 DC 0

* Current Source
Iq 6 9 DC

* Capacitors
Cc 6 7

* Resistors
R1 8 3 8k
Ro 7 6
Rl 7 5

* Transistors
M1 6 3 8 8 NMOS
Q2 6 2 5 QMODEL

* Model Definitions
.model NMOS NMOS
.model QMODEL NPN

* Simulation Commands
.tran 1n 10u
.end