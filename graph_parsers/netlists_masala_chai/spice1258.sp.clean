spice
* Netlist for the given schematic
VDD 3 0 DC 5V
V_in Vin 0 DC 1V
Vb Vb 0 DC 1.5V

* NMOS transistor: M1
M1 4 Vin 0 0 NMOS_MODEL

* PMOS transistor: M2
M2 2 Vb 4 4 PMOS_MODEL

* Resistors
RD1 3 4 1k
RD2 3 2 1k

.model NMOS_MODEL nmos
.model PMOS_MODEL pmos