spice
* Differential Pair Circuit

VCC 1 0 DC 15
VEE 5 0 DC -15

RC1 1 2 5k
RC2 4 3 5k
RE 2 5 7.5k

Q1 2 2 0 QMODEL
Q2 3 2 0 QMODEL

.model QMODEL NPN(IS=1E-16 BF=100)