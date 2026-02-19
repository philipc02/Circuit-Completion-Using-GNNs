spice
* Example SPICE netlist
* Op-Amp Inverting Amplifier

V1 vi 0 DC 0

R1 vi 4 1k
R2 4 2 1k
R3 2 6 1k
R4 2 4 1k

* Ideal Op-Amp
E1 3 0 4 5 1

* Voltage Output
Vout 3 vo DC 0

* Ground
Vgnd 5 0 DC 0

.ends