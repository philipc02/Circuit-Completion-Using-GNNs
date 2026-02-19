plaintext
* SPICE Netlist for the given schematic

* Transistors
Q1 2 6 3 NPN
Q2 3 7 4 NPN

* Resistors
RC 5 6 1k
RC 5 7 1k
RB 8 2 10k
RB 4 3 10k

* Current Source
IQ 2 3 DC 1mA

* Voltage Sources
V1 8 0 DC (vd/2)
V2 4 0 DC (-vd/2)

* Nodes
* 0 - Ground
* 5 - V+