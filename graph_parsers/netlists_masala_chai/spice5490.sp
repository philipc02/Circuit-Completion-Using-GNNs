spice
* Netlist for the given schematic

VDD 3 0 DC -10V
Vin 1 0

* Capacitor
C1 1 2 0.1u

* Resistors
RG 2 5 10Meg
RS 4 6 10k

* JFET
* The 2N4360 is a P-channel JFET, typically modeled with parameters such as Vto, Bf, etc.
XJFET 3 2 4 4 2N4360

* Connections
Vout 4 0

* Additional connections to ground
R1 5 0 0
R2 6 0 0

.END