plaintext
* SPICE Netlist

* Voltage Source
VCC 2 0 DC 20V
Vin 3 0

* Resistors
R1 2 3 390
R2 3 5 390
RL 4 0 16

* Capacitors
C1 3 2 10uF
C2 3 5 10uF
C3 4 6 100uF

* Diodes
D1 3 2 D_model
D2 3 5 D_model

* Transistors
Q1 2 3 1 QNPN
Q2 4 3 5 QNPN

.model QNPN NPN (IS=1E-14 BF=200 VAF=100)
.model D_model D (IS=1E-14)

.end