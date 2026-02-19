spice
* SPICE Netlist for the given circuit
VDD 1 0 DC 1.8V
RD 1 2 1k
RG 2 3 1k
M1 2 3 0 0 NMOS
* Model card for NMOS transistor
.model NMOS NMOS (Level=1 VTO=0.7 KP=120u)