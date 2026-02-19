spice
* SPICE Netlist for the given schematic

VPLUS 1 0 DC 30V
VMINUS 2 0 DC -30V
VS 9 0 DC 0V

R1_top 1 5 1k
R1_bottom 2 4 1k
RL 6 0 8

D1 5 4 D
D2 4 9 D

C_inf 9 4 CAP INFINITE

Qn 3 5 4 NMOS
Qp 4 3 6 PMOS

.model NMOS nmos
.model PMOS pmos
.model D D
.model CAP C

.end