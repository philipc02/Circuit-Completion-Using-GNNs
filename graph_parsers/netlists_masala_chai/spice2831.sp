* Netlist for the given schematic

V1 9 8 DC Vn_R1
R1 8 2 R1
M1 3 2 4 4 NMOS
R2 5 3 RD
R3 4 7 RS
V2 6 5 DC Vn_RD
V3 4 7 DC Vn_RS

* Node 3 is the drain, 2 is the gate, and 4 is the source and body for the NMOS