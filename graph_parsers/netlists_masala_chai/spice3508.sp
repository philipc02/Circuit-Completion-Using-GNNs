* SPICE Netlist

V1 2 6 DC 0
RT 2 7  R1
VIN 7 6  DC 0
RS 7 9 R2
IX1 9 3 DC 0
RL1 4 3 R3
C1 3 2 C1
C2 3 4 C2
IX2 5 6  DC 0
RL2 3 6 R4
ROUT 8 6 R5
VOUT 6 3 DC 0

.control
tran 1n 10n
plot v(3)
.endc
.end