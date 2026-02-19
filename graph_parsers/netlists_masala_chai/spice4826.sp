spice
* Transistor
Q1 2 7 4 QMOD

* Voltage Source
VCC 6 0 DC 10V
Vg 5 0 AC 1mV

* Resistors
R1 6 2 10k
R2 2 3 2.2k
R3 2 4 3.6k
R4 4 0 1k
R5 7 5 600
R6 2 7 10k
R7 2 0 10k

* Capacitors
C1 5 6 1u
C2 2 0 1u
C3 4 0 1u

* Models
.model QMOD NPN (BF=100)