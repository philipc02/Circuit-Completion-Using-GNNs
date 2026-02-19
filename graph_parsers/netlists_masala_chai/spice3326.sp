plaintext
* SPICE netlist for the provided schematic

* Voltage Sources
Vin 1 0 DC 0
VDD 5 0 DC supply_voltage

* Transistors
M1 2 1 4 4 NMOS
M2 2 3 5 5 PMOS

* Resistor
Rs 4 0 resistor_value

* Node Mapping
* Node 1: Vin
* Node 2: Vout
* Node 3: Gate of M2
* Node 4: Source of M1 / Resistor Rs
* Node 5: VDD

.end