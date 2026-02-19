* Sample SPICE Netlist for the Given Schematic

* Voltage Source
V1 2 3 DC 0

* Operational Amplifier
* Vin1 (non-inverting) = Node 2
* Vin2 (inverting) = Node 3
* Vout = Node 4
A0 2 3 4 OPAMP

* Resistors
R1 4 2 1k
R2 2 3 1k

* Note: Node numbers are based on the annotated schematic.