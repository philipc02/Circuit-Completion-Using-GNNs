spice
* Diode
D1 3 4 Dmodel

* Capacitor
C1 4 5 Cvalue

* Resistor
R1 4 2 Rvalue

* Voltage Source
V1 3 5 DC Vvalue

.model Dmodel D(IS=1e-14)
.end