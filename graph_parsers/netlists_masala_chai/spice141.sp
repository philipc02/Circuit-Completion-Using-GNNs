* SPICE Netlist
* Components
I3 2 1 DC i_3
J3 2 5 DC (1 - e_m)j_3
I2 5 6 DC i_2
Itail 8 9 DC i_tail

V1 8 0 DC V1
V2 4 0 DC V2

R1 2 3 Rm(i1/r)
R2 3 0 ro(dp)
R3 6 7 Re(i)
R4 7 0 ro(dp)
R5 5 6 ro(dn)
R6 1 5 Rm(j3)

* Connections
* 1 to ground, based on node 9
* Connect output node 6 to i_out