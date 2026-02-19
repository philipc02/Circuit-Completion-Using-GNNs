spice
* SPICE Netlist for the given circuit

V1 6 0 DC VCM
V2 7 0 DC VCM
IQ 8 2 DC I_Q
RC1 9 5 RC
RC2 9 5 RC
RB1 6 2 RB
RB2 7 2 RB
Q1 5 6 2 NPN
Q2 5 7 2 NPN

* Global nodes
V+ 9 0 DC V+
V- 8 0 DC V-

.END