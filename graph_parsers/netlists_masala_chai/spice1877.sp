spice
* SPICE netlist for the given schematic

VDD 3 0 DC 5

* Current Sources
I1 3 2 DC 1m
Iin 4 0 DC 1m

* PMOS Transistor - M2
* Drain - 3, Gate - 2, Source - 3
M2 3 2 3 PMOS_MODEL

* NMOS Transistor - M1
* Drain - 2, Gate - 4, Source - 0
M1 2 4 0 NMOS_MODEL

* Voltage Output
Vout 2 0

.model PMOS_MODEL PMOS
.model NMOS_MODEL NMOS

.end