* SPICE netlist for the given circuit
VCC 4 0 DC 12V
VG 8 0 SINE(0 1 1k)

RG 8 7 1k
R1 4 7 10k
R2 7 2 10k
RE 5 2 1k
RC 4 5 5k
RL 3 0 5k

Cin 7 5 1u
Cout 5 3 1u
CE 5 2 1u
Cc 5 2 10p
Cstray 5 3 5p

Q1 5 7 2 NPN

.END