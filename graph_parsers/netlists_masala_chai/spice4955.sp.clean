spice
* SPICE netlist for given schematic

* Voltage Source
VDD 2 0 DC 30

* Resistors
R1 7 2 2meg
R2 3 0 1meg
RS 6 0 2.2k
RL 5 6 3.3k

* Capacitor
Cin 7 2 <Value> ; value to be determined based on requirement

* MOSFET (N-channel)
M1 2 7 6 6 NMOS L=1u W=1u

* Input Voltage Source
Vin 7 0 AC <Value> ; specify AC signal details

* .MODEL declaration assumed for NMOS (example)
.model NMOS NMOS level=1

.end