plaintext
* SPICE netlist for BJT Amplifier Circuit

Vg 7 0 AC 1m
Vcc 4 0 DC 10

RG 3 7 50

RE 8 0 2.2k
R1 9 2 10k
R2 8 0 2.2k
RC 2 4 3.6k
RL 2 11 10k

C1 9 7 47uF
C2 8 2 47uF
C3 2 0 1uF

Q1 2 9 8 QNPN

.model QNPN NPN (IS=1E-14 BF=100)

.end