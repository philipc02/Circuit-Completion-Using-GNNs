plaintext
* SPICE netlist for the given schematic
V1 4 0 DC 1V ; Input voltage source Vi
RC 1 3 1k ; Resistor connected to VCC and collector
Q1 3 5 0 QNL ; NPN Transistor, 3=collector, 5=base, 0=emitter
VCC 1 0 DC 10V ; VCC supply voltage