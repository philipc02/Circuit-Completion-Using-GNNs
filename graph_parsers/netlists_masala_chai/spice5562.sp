* SPICE Netlist
* Components

R1 2 3 9k
R2 7 2 1k
R3 1 4 2k
R4 8 4 3k

* Ground and voltage sources
V1 1 0 DC v1
V2 8 0 DC v2

* Operational Amplifier
* Node 2: Inverting Input, Node 4: Non-Inverting Input, Node 3: Output
X1 2 4 3 OPAMP

* Define nodes
.node 0

* End of netlist