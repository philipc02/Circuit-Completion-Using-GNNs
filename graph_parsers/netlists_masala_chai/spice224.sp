spice
* SPICE netlist for the given schematic

VCC 6 0 DC VCC
RC1 6 3 1k ; Resistor RC1 between nodes 6 and 3
RC2 3 3 1k ; Resistor RC2 at node 3
Q1 6 5 4 NPN ; NPN transistor Q1 (C=B=E configuration)
Q2 3 2 4 NPN ; NPN transistor Q2 (C=B=E configuration)
IB1 5 0 DC 0 ; Base current source for Q1 to node 5
IB2 2 0 DC 0 ; Base current source for Q2 to node 2
IEE 4 0 DC IEE ; Current source IEE from node 4 to ground