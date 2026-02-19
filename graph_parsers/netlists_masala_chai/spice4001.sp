plaintext
* SPICE netlist for BJT Amplifier Circuit

V1 5 0 DC 4V
V2 7 0 DC -6V
Vs 1 5 DC 0V

RS 1 2 1k
RB 2 3 5k
RE 9 6 5k
RC 9 7 4k
RL 6 8 4k

CC1 2 3
CC2 6 8
CE 6 7

Q1 6 3 9 QNPN

.model QNPN NPN (Is=1e-14 Bf=100 Vaf=100V)

.end