spice
* SPICE netlist for the given schematic

Q1 2 5 3 NPN
Q2 2 5 3 NPN

I1 3 0 DC 1mA

RC1 2 4 5k
RC2 2 4 5k
R_Var 2 2 1k

VCC 4 0 DC VCC

.control
.endc

.end