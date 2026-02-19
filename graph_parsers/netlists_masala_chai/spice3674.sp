plaintext
* SPICE Netlist

VI 1 5 DC 0
R1 1 5 R
D1 2 5 D_model
D2 5 6 D_model
V2 3 4 DC 0

* Models
.model D_model D

.end