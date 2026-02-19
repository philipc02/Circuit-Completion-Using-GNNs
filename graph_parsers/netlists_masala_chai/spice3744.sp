spice
* SPICE netlist
* VCCS: Voltage-controlled current source
G1 3 4 4 8 gm

* Resistor
R1 2 3 ro

* PMOS Transistor
* Mname drain gate source body model
M1 3 8 4 4 PMOS_Model

* Node assignments
* 1 - n/c here, nodes are given from image
* 2 - Common Source
* 3 - VCCS Output / Drain
* 4 - Source
* 5, 6, 7 - n/c in this part
* 8 - Gate
.end