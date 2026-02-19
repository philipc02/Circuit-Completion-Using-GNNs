spice
* Transistors
Q1 2 3 1 Q_NPN
Q2 5 2 2 Q_PNP

* Current Source
I1 6 3 DC IBIAS

* Diodes
D1 3 1 D
D2 1 2 D

* Resistor
RL 2 4 RL_value

* Voltage Sources
VCC 3 0 DC VCC_value
VEE 5 0 DC -VCC_value

* Nodes
* 1: base of Q_N and Q_P
* 2: collector of Q_N and output
* 3: collector of IBIAS and base of Q_N
* 4: ground for RL
* 5: collector of Q_P
* 6: IBIAS

.end