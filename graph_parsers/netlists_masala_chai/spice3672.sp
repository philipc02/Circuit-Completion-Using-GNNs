* Diode Clamp Circuit Netlist

V1 1 0 DC 5V
V2 3 0 DC 5V

R1 1 2 10k
R2 4 5 10k

D1 2 1 Dmodel
D2 2 3 Dmodel
D3 4 5 Dmodel
D4 5 3 Dmodel

.model Dmodel D

.end