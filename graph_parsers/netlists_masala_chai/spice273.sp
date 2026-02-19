// SPICE netlist for the schematic
* Voltage and resistor definitions
V1 7 8 DC 0

* Resistors
R1 7 6 1k
R2 6 3 1M
R3 8 5 1k
R4 5 4 1M

* Operational Amplifier
XOP 2 4 3 OPAMP

* Ground
.model OPAMP opamp