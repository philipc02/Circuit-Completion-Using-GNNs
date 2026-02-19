spice
* SPICE Netlist for the given schematic

VDD 2 0 DC 5V

* PMOS Transistor
ML 4 2 2 2 PMOS_MODEL

* NMOS Transistors
MDA 5 5 3 3 NMOS_MODEL
MDB 4 4 3 3 NMOS_MODEL
MDC 3 3 0 0 NMOS_MODEL

* Model definitions (example)
.model PMOS_MODEL PMOS (kp=1u Vto=-0.7)
.model NMOS_MODEL NMOS (kp=2u Vto=0.7)

* End of netlist