plaintext
* Transistor Q1
Q1 6 9 5 QNPN

* Transistor Q2
Q2 4 8 2 QNPN

* Transistor Q3
Q3 7 7 2 QNPN

* Resistors
R1 2 7 R1_VALUE
R2 4 2 R2_VALUE
R3 9 7 R3_VALUE
RL 4 10 RL_VALUE

* Voltage Sources
Vi 1 2 DC Vi_VALUE
VCC 5 0 DC VCC_VALUE
VEE 2 0 DC -VCC_VALUE

* Current Source
IRL 10 0 DC IRL_VALUE

.model QNPN NPN