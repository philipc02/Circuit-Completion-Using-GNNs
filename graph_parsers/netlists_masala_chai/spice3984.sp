plaintext
* SPICE Netlist for the Given Schematic

V1 8 3 DC 0
R1 9 2 1k
R2 4 3 1k
RC 2 6 1k
CC 8 5 1uF
Q1 6 5 3 NPN
VCC 9 0 DC 12

* Connections
* V1 (Vs) connects between nodes 8 and 3
* CC connects between nodes 8 and 5
* R1 connects between node 9 (VCC) and node 2
* R2 connects between node 4 and node 3 (GND)
* RC connects between nodes 2 and 6 (collector of Q1)
* Q1 Collector at node 2, Base at node 5, Emitter at node 3 (NPN)

.model Q1 NPN (BF=100)
.end