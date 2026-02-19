* SPICE Netlist

M1 2 3 6 6 NMOS
M2 4 2 6 6 NMOS
M3 7 3 5 5 PMOS
M4 5 2 5 5 PMOS

I1 6 0 Iss

VDD 5 0 DC VDD

* Node Mapping
* 1 : X (Internal node, directly mapped as X)
* 2 : Vout
* 3 : Gate of M1 and M3
* 4 : Output (Vout)
* 5 : VDD
* 6 : Ground
* 7 : Gate of M3

.END