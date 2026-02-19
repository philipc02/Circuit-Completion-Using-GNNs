plaintext
* SPICE Netlist for the given schematic

** Voltage Sources
V1 6 0 DC -2V
VEE 4 0 DC -5.2V

** Transistors
Q1 6 6 0 NPN
QA 2 4 4 NPN
QR 2 4 4 NPN
Q2 5 5 0 NPN

** Resistors
RC1 2 6 220
RC2 3 5 245
RT1 6 0 50
RT2 5 0 50
RB 2 4 50k
RE 4 0 779

** Voltage Reference
VR 2 5 DC -1.32V

** End of Netlist