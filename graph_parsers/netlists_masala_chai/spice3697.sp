* SPICE Netlist
V1 8 0 DC -5V
V2 6 0 DC 5V
R1 6 3  R1_value
R2 5 3  R2_value
R3 5 4  R3_value
RD 3 7  RD_value
RS 4 2  RS_value
CG 3 0  CG_value
CC 4 5  CC_value
CS 2 0  CS_value
M1 3 5 4 4 NMOS_model
M2 7 3 3 3 NMOS_model
VI 1 8 DC some_value
.end