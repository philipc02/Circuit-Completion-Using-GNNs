spice
* Example SPICE Netlist

M1 3 2 6 6 NMOS
M2 4 5 3 3 PMOS
RD 4 VDD 1k ; replace 1k with the actual resistance value
VDD VDD 0 DC 5V  ; replace 5V with the actual DC voltage value
Vin 2 6 DC 0V AC 1V ; replace AC amplitude with the actual signal
Vb 5 0 DC 1.2V ; replace 1.2V with the actual bias voltage

* Define models for NMOS and PMOS if needed
.model NMOS NMOS (Level=1)
.model PMOS PMOS (Level=1)

.end