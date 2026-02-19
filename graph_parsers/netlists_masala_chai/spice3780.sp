spice
* Netlist for the given schematic

* Voltage Source
Vi 6 5 DC 0

* Resistors
R1 3 9 95.4k
R2 3 4 150k
R3 6 5 54.6k
RD 8 9 2.5k
RS 6 2 10k

* Capacitors
CG 4 6
CC 4 5
CS 2 0

* MOSFETs
* M1 is NMOS (Drain=5, Gate=4, Source=6)
M1 5 4 6 6 NMOS
* M2 is PMOS (Drain=8, Gate=3, Source=9)
M2 8 3 9 9 PMOS

* Power Supplies
V+ 9 0 DC 5V
V- 2 0 DC -5V

* The output node is connected at vo
* The input node for Vi is at node 6