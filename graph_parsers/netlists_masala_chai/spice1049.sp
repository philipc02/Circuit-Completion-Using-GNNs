spice
* SPICE netlist for the schematic

Q1 3 2 4 NPN
Q3 2 3 5 PNP

RC 2 3 1k
RE 4 0 1k

VCC 5 0 DC 10V
Vin 3 0 DC 2V
Vb 2 0 DC 2V

.model NPN NPN
.model PNP PNP

.end