spice
* Simple NPN transistor circuit

Vcc 1 2 DC 3V
Vi 4 5 DC 0V
I1 2 0 DC 0.2mA
R1 1 2 10k
C1 5 2
Q1 2 5 3 QNPN

.model QNPN NPN (IS=1E-14 BF=100)

.tran 10u 1m
.end