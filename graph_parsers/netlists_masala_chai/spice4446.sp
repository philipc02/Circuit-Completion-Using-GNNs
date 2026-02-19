* SPICE netlist for the given schematic

* Voltage Source
Vs 3 0 DC

* Current Source
Io 6 4

* Resistors
RC1 6 3 5k
RC2 6 2 2k
RC3 6 5 1k
RE1 3 7 0.1k
RE2 5 0 0.1k
RF 4 5 0.8k

* Transistors - BJT NPN
Q1 6 3 7 QNL
Q2 4 3 2 QNL
Q3 6 4 5 QNL

.model QNL NPN