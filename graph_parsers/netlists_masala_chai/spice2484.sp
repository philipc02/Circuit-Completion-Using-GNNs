* Netlist for the given schematic

VDD 3 0 DC 5V

* NMOS M1
M1 2 Vin 3 3 NMOS

* NMOS M2
M2 0 Vb 2 0 NMOS

* Model Parameters (assumed for demonstration)
.model NMOS NMOS (Level=1 Vto=1.0 KP=50u)

.END