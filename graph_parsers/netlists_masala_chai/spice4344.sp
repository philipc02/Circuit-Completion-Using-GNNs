spice
* BJT Differential Pair Circuit

Q1 6 9 8 QNPN
Q2 7 4 8 QNPN

RE 8 V+ RE_VALUE
RC1 6 3 RC_VALUE
RC2 7 2 RC_VALUE

V1 v1 9 DC 0
V2 v2 4 DC 0
V3 8 V+ DC VPOSITIVE
V4 5 V- DC 0

* Models
.model QNPN NPN(Is=1e-16 Vaf=100 Bf=100)
* Parameters
* Replace RE_VALUE, RC_VALUE, VPOSITIVE with actual values