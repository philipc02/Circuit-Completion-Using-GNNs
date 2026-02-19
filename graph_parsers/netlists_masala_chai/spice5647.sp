plaintext
* SPICE netlist

V1 vi 0 DC 0
R1 vi 2 1k
R2 2 4 1k
D1 4 3 D
V3 5 0 DC 0

* Model definition for diode (can be any appropriate diode model)
.model D D

.end