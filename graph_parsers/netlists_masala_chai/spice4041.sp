spice
* NMOS amplifier circuit

* Voltage sources
Vi 4 0 DC 0
Vp 2 0 DC 5
Vn 6 0 DC -5

* Resistors
RD 2 3 6.7k
RG 4 7 50k
RS 6 5 5k
RL 3 0 10k

* Capacitor
CC 2 3

* NMOS transistor
M1 2 7 5 5 NMOS

* Model definition for NMOS (assuming a generic model)
.model NMOS NMOS (Level=1 Vto=1.5)

.end