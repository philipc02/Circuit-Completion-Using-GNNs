plaintext
* Netlist for the provided schematic

* Voltage source
V1 3 0 DC Vi

* Resistor
R1 3 2 100k

* MOSFET (Assumed to be an NMOS)
M1 4 3 5 5 NMOS

* Current source
I1 5 0 DC I

* Coupling capacitor
C1 4 2 Cc

* Load resistor
R2 2 0 Ro

* Load resistor
R3 2 6 RL

* Power supply connections
VDD 4 0 DC 9
VSS 5 0 DC -9

.model NMOS nmos
.ends