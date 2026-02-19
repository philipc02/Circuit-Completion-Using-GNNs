plaintext
* Differential Amplifier Circuit

VCC 8 0 DC 10V

* Transistors
Q1 2 2 5 QNPN
Q2 4 3 5 QNPN

* Resistors
RC1 6 8 10k
RC2 3 8 10k
RL 2 4 20k
RE 5 3 300

* Current Sources
I1 2 9 DC 0.5m
I2 4 11 DC 0.5m

R1 9 0 200k
R2 11 0 200k

.model QNPN NPN (IS=1E-14 BF=100)

.END