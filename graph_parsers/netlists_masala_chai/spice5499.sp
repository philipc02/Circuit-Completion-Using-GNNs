plaintext
* SPICE netlist for the BJT amplifier circuit
VCC 1 0 DC 10
VIN 4 0 DC 1

RB 4 2 22k
RC 1 6 1k

Q1 6 2 3 NPN bf=50

* Connections:
* VCC is connected to node 1
* VIN is the input voltage at node 4
* RB, the base resistor, connects node 4 to base of the transistor at node 2
* RC, the collector resistor, connects node 1 to the collector of the transistor at node 6
* Transistor Q1 has its collector at node 6, base at node 2, and emitter at node 3 (ground)