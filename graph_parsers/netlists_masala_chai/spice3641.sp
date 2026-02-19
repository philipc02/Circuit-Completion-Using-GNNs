* SPICE Netlist
V1 1 0 DC 0
I1 3 5 DC 0
R1 1 5 10k
R2 5 4 10k
D1 5 0 D_Zener
D2 5 2 D_Diode

.model D_Zener D(Rs=0.5 Vj=0.7)
.model D_Diode D

.end