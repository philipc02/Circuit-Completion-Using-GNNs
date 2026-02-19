plaintext
* SPICE Netlist

V1 9 0 DC
V2 7 6 DC 1.6
R_E1 4 5 2.6k
R_E2 5 0 1.2k
R_C 6 7 0.8k

Q1 3 4 5 NPN
Q2 5 4 0 NPN
Q3 8 6 5 NPN

* Voltage source for top rail
VCC 4 0 DC 5

* Connections for inputs and additional nodes (if any)