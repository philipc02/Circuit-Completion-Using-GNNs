spice
* SPICE Netlist for the circuit
VDD 4 0 DC 5V
Vb1 3 0 DC 2.5V
Vb2 4 0 DC 2.5V
Vin 1 0 DC 0V

* NMOS M1: Drain Gate Source
M1 2 3 2 NMOS_MODEL

* PMOS M2: Drain Gate Source
M2 2 4 4 PMOS_MODEL

* Resistor
RS 2 1 1k

* Define models for NMOS and PMOS
.model NMOS_MODEL NMOS level=1
.model PMOS_MODEL PMOS level=1

.end