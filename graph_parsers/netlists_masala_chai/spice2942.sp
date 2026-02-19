spice
* SPICE netlist generated from schematics

* Voltage source
Vt 6 7 DC [Value]

* NMOS transistor
M1 2 3 4 4 NMOS_MODEL

* Capacitors
C1 5 7 [Value_C1]
C2 3 7 [Value_C2]

* Current source
I1 4 7 DC [Value]

* Resistor
RD 2 VDD [Value_RD]

* Model definitions (replace with actual models)
.MODEL NMOS_MODEL NMOS (LEVEL=1)

* End of netlist