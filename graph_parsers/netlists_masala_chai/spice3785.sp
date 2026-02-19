spice
* SPICE Netlist

V1 3 0 DC vi
V2 2 0 DC +10
V3 5 0 DC -10

C1 3 2 10uF
C2 2 4 10uF

RG 2 0 50k
RS 5 4 10k
RL 4 6 10k

M1 2 2 5 5 NMOS

*.model NMOS NMOS(Level=?)
.end