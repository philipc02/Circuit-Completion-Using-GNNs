plaintext
* SPICE Netlist
V1 2 0 DC 0.4V
V2 4 0 DC 0.6V
I1 C 3 DC {beta*is}
D1 3 2 D1_model
D2 3 4 D2_model

.model D1_model D
.model D2_model D

* Node Definitions
* C = 1
* E = 0
* B = Not used in the netlist; External bias (i_s) introduced