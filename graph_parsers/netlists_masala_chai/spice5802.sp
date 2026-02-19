* SPICE Netlist

* Voltage Source
V1 1 4 DC

* Resistor RG
RG 1 2 100k ; replace 100k with the actual value if known

* Voltage-Controlled Current Source
G1 3 4 VALUE = { gm * V(1, 3) } 

* Resistor RL'
RL 5 3 10k ; replace 10k with the actual value if known

* Ground
V0 3 0 0