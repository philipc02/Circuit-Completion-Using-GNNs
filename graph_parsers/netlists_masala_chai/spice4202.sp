spice
* Example SPICE netlist generated from schematic

Vt 3 0 DC 0
R1 6 3 R1_value
R2 3 2 R2_value
R3 2 4 R_value
R4 4 0 R_value

* Op-Amp models
* Assuming ideal op-amps for simplicity
* First Op-Amp
XU1 3 6 2 opamp
* Second Op-Amp
XU2 0 2 4 opamp

.model opamp opamp