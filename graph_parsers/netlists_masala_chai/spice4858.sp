spice
* SPICE Netlist for BJT Amplifier

VCC 3 0 DC 15V
vg 7 0 AC 1V

RG 6 7 600
C1 6 2 1uF
R1 3 5 10k
R2 2 0 20k
RE 2 4 60
RL 2 0 30

Q1 5 6 2 QNPN
Q2 8 5 2 QNPN

.model QNPN NPN (IS=1E-14 BF=100)

.end