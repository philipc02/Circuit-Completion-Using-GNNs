plaintext
* SPICE Netlist for the Circuit

* Voltage Sources
VCC 7 0 DC 10
Vin 3 0 DC 5

* Resistors
RC 6 7 1k
RM 5 0 100

* Transistors (assuming NPN for Q1 and Q2)
Q1 5 3 7 NPN
Q2 2 3 6 NPN

* Diode
D1 2 5 Laser

* Detailed SPICE notations might be needed for real components
.model NPN NPN
.model Laser D

.end