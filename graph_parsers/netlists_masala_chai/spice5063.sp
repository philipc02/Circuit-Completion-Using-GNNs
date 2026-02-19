plaintext
* Differential Amplifier Circuit

Q1 3 2 7 NPN
Q2 3 6 7 NPN

RC1 5 3 7.5k
RC2 5 3 7.5k
RE 7 4 7.5k
RL 3 3

VCC 5 0 DC 15V
VEE 4 0 DC -15V
V1 1 0 AC 10mV

* Node 0 is the ground reference
* NPN model
.model NPN npn (Is=1e-14 Bf=100)
.end