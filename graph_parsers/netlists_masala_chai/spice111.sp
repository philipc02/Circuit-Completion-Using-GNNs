plaintext
* Differential amplifier
VCC 8 0 DC VCC
VEE 2 0 DC -VEE

* Resistors
RC1 8 5 RC
RL 5 4 RL
RC2 8 3 RC
RE 6 2 RE

* Current Sources
IEE1 6 0 DC IEE
IEE2 3 0 DC IEE

* Transistors
Q1 5 9 6 QMOD
Q2 4 6 3 QMOD

* Model for NPN BJTs
.model QMOD NPN (IS=1e-14 BF=100)