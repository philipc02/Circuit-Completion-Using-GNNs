plaintext
* SPICE Netlist for the given schematic

VDD 5 0 DC 1.8V

M_REF 5 3 3 3 PMOS_MODEL
M2 5 3 2 2 PMOS_MODEL
M1 2 6 0 0 NMOS_MODEL

I_REF 3 0 DC 1mA

* Voltage source for input
Vin 6 0 DC 0.0V

.model PMOS_MODEL PMOS
.model NMOS_MODEL NMOS

.end