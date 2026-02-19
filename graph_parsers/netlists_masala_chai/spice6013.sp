* Differential Amplifier SPICE Netlist

Q1 3 1 2 NPN
Q2 4 5 7 NPN

RC1 3 1 RC
RC2 4 5 RC

I1 6 2 DC I

VCC 1 0 DC VCC
VEE 6 0 DC VEE

*.model NPN NPN (IS=1e-14 BF=100)
.end