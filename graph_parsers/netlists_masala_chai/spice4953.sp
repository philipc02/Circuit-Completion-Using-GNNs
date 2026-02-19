spice
* SPICE Netlist
Vg 8 4 AC 1m
Vdd 0 5 DC 20

RG 8 2 47k
R1 5 7 10Meg
R2 7 3 10Meg
RS 3 2 1k
RL 33 3 1k

M1 32 7 3 3 NMOS

.include 'nmos.model'