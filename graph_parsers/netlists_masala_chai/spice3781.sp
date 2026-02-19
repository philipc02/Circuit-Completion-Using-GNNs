spice
* SPICE Netlist for the given schematic
V1 5 4 DC Vi
R1 5 2 R3_par_R2
R2 2 4 RD
E1 2 7 2 2 gm1
E2 2 Vo 2 2 gm2
RL Vo 0 1e6
.END