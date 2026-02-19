spice
* SPICE netlist for Op-amp circuit with resistors and voltage sources

* Voltage Sources
V0 4 0 DC <V0>
V1 0 1 DC <V1>
V2 0 2 DC <V2>
V3 0 3 DC <V3>

* Resistors
R1 3 2 R
R2 3 5 2R
R3 3 6 4R
R4 4 3 8R
R5 2 5 R

* Op-amp
* Assuming ideal op-amp with nodes 3 and 5 as inputs and node 5 as output
E1 5 0 3 0 1MEG
GND 0

.END