plaintext
* Example SPICE Netlist
V1 6 0 DC 5V
R1 3 4 2k
R2 5 6 2k
D1 2 3 D
D2 4 5 D
* Define model for diodes
.model D D
.end