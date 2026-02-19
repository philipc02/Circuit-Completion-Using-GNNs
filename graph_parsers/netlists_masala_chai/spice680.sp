plaintext
* SPICE Netlist
Iin 3 4 DC 0
R1 3 2 R1_value
D1 3 2 D1_model
VB 2 0 DC VB_value

* Define models and values here:
.model D1_model D
.PARAM R1_value=100
.PARAM VB_value=5

.END