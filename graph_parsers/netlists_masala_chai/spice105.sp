spice
* SPICE Netlist for the given schematic

* Voltage Sources
VCC 3 0 DC 10V
Vi 4 0 DC 5V

* Transistors
Q1 3 4 5 NPN
Q2 3 3 6 NPN

* Resistors
R1 2 5 1k
R2 6 0 50

* Ground reference
V0 0 0 DC 0V

* Simulation commands
.TRAN 0.1ms 10ms
.END