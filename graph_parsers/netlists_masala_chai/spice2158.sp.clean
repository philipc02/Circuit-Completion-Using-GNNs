spice
* NMOS Amplifier Circuit

* Voltage Source
VDD 2 0 DC VDD_VALUE

* Resistors
RD 2 drain_node RD_VALUE
RS source_node 0 RS_VALUE

* NMOS Transistor
M1 drain_node gate_node source_node source_node NMOS_MODEL

* Connections
* Drain of M1 connects to RD
* Gate of M1 is connected to ground (gate_node is 0)
* Source of M1 connects to RS and ground

* Voltage Source VDD connects to RD

* Define nodes
drain_node 1
gate_node 0
source_node 3

* .model NMOS example
.model NMOS_MODEL NMOS (LEVEL=1)

.end