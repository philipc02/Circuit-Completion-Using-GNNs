spice
* SPICE netlist for the given schematic

* Voltage Source
VPS 8 5 DC <voltage_value>

* Input Resistor
Ri 2 8 <resistance_value>

* Zener Diode
D1 5 5 DZENER

* Load Resistor
RL 4 3 <resistance_value>

* Model for the Zener Diode
.model DZENER D(BV=<zener_voltage_value>)

.end