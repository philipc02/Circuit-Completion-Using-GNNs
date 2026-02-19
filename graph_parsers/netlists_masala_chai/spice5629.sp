plaintext
* SPICE Netlist

R1 1 5 10k
R2 2 7 10k
R3 3 4 10k
R4 5 4 10k

V1 7 4 DC 5
V2 3 4 DC 5

D1 8 1 DMOD
D2 6 2 DMOD

.model DMOD D

* Node Definitions
* Node 1: Top-left
* Node 2: Between D2, R2
* Node 3: Between V2, R3
* Node 4: Common ground
* Node 5: Right side, above R4
* Node 6: Between D2 and V2
* Node 7: Between V1, R2
* Node 8: Between D1, V1

.end