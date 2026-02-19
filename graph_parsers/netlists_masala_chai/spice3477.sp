spice
* SPICE netlist for the given circuit

VCC 4 0 DC [VALUE]
VEE 6 0 DC -[VALUE]

R1 4 2 [VALUE]
R2 5 3 [VALUE]

Q1 2 2 5 QMODEL
Q2 3 2 5 QMODEL

* Note: Replace [VALUE] with actual resistance and voltage values.
* QMODEL should be defined based on the specific BJT model used.