spice
* Circuit Description
Vg 7 0 DC 0 AC 1mV
Vcc 5 0 DC 10V

* Resistors
Rg 7 2 600
R1 2 4 10k
RC1 5 2 3.6k
R2 4 9 2.2k
RE1 4 0 1k
R3 5 6 15k
RE2 6 8 39k
R4 6 8 4.3k
RL 6 0 10k

* Capacitors
C1 2 0 C1_value
C2 5 6 C2_value
C3 4 0 C3_value

* NPN Transistors
Q1 4 3 9 QNPN
Q2 6 5 8 QNPN

.model QNPN NPN (IS=1E-14 BF=100)

.end