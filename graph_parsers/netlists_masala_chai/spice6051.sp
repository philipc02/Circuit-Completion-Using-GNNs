spice
* SPICE Netlist for the given schematic

* Q1 npn transistor
Q1 3 2 5 npn

* Q2 npn transistor
Q2 4 8 6 npn

* Resistors
R1 3 1 RD
R2 4 1 RD
R3 5 6 RS

* Current Sources
I1 7 3 DC 0.5
I2 6 3 DC 0.5

* Voltage Sources
VDD 1 0 DC VDD
VSS 0 0 DC VSS

* End of Netlist