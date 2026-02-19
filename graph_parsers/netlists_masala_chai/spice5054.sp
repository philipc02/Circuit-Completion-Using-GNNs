plaintext
* Differential Pair Amplifier Netlist
* Nodes: 1(Ground), 2, 3, 4, 5, 6

VCC 3 0 DC VCC
VEE 1 2 DC -VEE

Q1 3 Vin 5 NPN
Q2 4 Vin 6 NPN

RC1 3 4 RC
RC2 3 NodeConnectingRightRC RC

RE1 5 2 2RE
RE2 6 2 2RE

Vin Vin+ Vin- DC 0

* .model definitions
.model NPN NPN
.model PNP PNP

.end