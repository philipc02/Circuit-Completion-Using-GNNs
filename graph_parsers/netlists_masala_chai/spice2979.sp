* List of components from the schematic:
* C1, C2: Capacitors
* M1: NMOS transistor
* I1: Current source
* V1: Voltage source
* RD: Resistor

* SPICE netlist:

C1 7 3 <value_of_C1>
C2 7 2 <value_of_C2>
V1 3 2 <value_of_V1>

* NMOS transistor M1 with drain (4), gate (6), and source (2)
M1 4 6 2 2 NMOS_MODEL

I1 4 5 <value_of_I1>
RD 5 4 <value_of_RD>
VDD 5 0 <value_of_VDD>

* Define models
.model NMOS_MODEL NMOS(...)

* End of netlist