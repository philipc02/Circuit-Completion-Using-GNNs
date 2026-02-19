* SPICE Netlist

* Voltage Source
V1 5 0 DC 5V

* Input Voltage
Vi 1 0 DC 0

* Transistors
Q1 5 3 2 QMOD
Q2 2 7 6 QMOD

* Current Source
I1 2 0 DC 0.2mA

* Resistors
R1 3 2 250
R2 2 7 250
R3 7 0 25k

* Transistor Model
.model QMOD NPN (IS=1E-14 BF=100)

.end