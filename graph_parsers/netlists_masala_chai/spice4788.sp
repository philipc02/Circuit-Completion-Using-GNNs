spice
* Example SPICE netlist for NPN BJT Amplifier
VCC 3 0 DC 12V
R1 5 4 150
R2 4 0 33
RC 3 7 39
RE 6 2 10
Q1 7 4 6 NPN
.tran 1n 1u
.end