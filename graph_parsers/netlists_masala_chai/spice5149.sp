spice
* SPICE Netlist for the given schematic

Vin in 0 DC 0
R1 in 3 220
R2 2 3 43k
C1 2 2 100p

* Operational Amplifier (Ideal)
* - input connected to node 4
* + input connected to node 3
* output connected to node 2
XOP 3 4 2 OPAMP

.END