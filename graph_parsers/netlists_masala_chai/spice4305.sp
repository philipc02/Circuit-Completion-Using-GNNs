* Transistors
Q1 9 7 8 QNPN
Q2 8 10 3 QNPN
Q3 2 5 2 QNPN
Q4 4 12 2 QNPN

* Current Sources
I1 7 6 DC 0
I2 10 6 DC 0

* Resistors
RC1 7 6 8k
RC2 10 6 8k
R1 5 11 18.6k

* Power Supply
VCC 6 0 DC 5
VSS 4 0 DC -5

* Model Definitions
.model QNPN NPN (IS=1E-14 BF=100)

.end