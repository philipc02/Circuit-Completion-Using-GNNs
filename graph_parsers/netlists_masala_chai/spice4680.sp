spice
* SPICE Netlist
V1 2 3 Vp
L1 2 3 0
D1 2 4 D_SH
D2 4 3 D_PIV

.model D_SH D(IS=1e-14) 
.model D_PIV D(IS=1e-14)

.end