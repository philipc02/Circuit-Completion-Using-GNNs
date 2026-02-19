spice
* SPICE Netlist

V1 4 2 DC V+
R1 4 2 R
D1 2 3 D
RL 2 3 100k ; Load resistor to make the output node well-defined
.model D D