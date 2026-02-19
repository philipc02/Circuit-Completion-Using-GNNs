plaintext
*SPICE netlist for the given circuit

V1 3 0 DC 5V
I1 4 0 DC Is

Q1 8 4 6 NPN

RF 3 2 10k
RB 4 7 100k
RC 8 2 1k
RE 6 9 0.5k

CC1 4 0 inf
CC2 5 2 inf
CE 6 9 inf

.tran 1u 10m
.end