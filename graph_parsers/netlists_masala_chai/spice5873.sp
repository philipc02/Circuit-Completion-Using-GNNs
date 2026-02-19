* SPICE Netlist
V1 3 0 DC 10
MN1 2 3 6 6 NMOS
R1 3 3 3k
R2 3 7 2k
RS 5 2 5k
RD 2 6 4k
VG 3 0 DC 0
VS 5 0
VD 2 0
.model NMOS NMOS (Level=1)
.end