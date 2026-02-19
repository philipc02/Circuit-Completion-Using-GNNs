plaintext
* SPICE netlist for the given schematic

V1 1 0 DC 12V
R1 1 2 30k
R2 2 0 10k
D1 2 3 DIODE_MODEL
R3 3 0 5k

.model DIODE_MODEL D
.end