spice
* BJT Amplifier Circuit
VCC 5 0 DC 30
Vg 4 0 AC 0.5

RG 4 3 600
R1 5 8 10k
R2 3 2 10k
RE 2 9 100
RL 9 10 100

C1 3 6
C2 9 10

Q1 5 3 2 BJT

.model BJT NPN (BF=300)
.tran 0.1ms 10ms

.end