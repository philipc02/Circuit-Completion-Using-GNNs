plaintext
* SPICE Netlist for the Differential Amplifier Circuit

VCC 6 0 DC VCC
VEE 2 0 DC -VEE

V1 3 0 DC V1
V2 4 0 DC V2

RC1 6 5 RC
RC2 6 8 RC
RL 5 8 RL
RE 2 0 RE

Q1 5 3 2 NPN
Q2 8 4 2 NPN

.model NPN NPN (IS=1e-14 BF=100)

.end