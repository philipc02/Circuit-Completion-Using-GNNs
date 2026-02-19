spice
* SPICE Netlist for the given schematic

* Current Source
I1 3 2 DC 0.2mA

* Voltage Source
V1 29 0 DC [Input Voltage as Required]

* NMOS Transistor Q1
M1 2 2 7 7 NMOS

* NMOS Transistor Q2
M2 6 2 4 4 NMOS

* Resistors
R1 4 0 3.5k
R2 2 7 14k

* Analysis
.dc V1 0 5 0.1
.end