plaintext
* SPICE netlist for the schematic

* Voltage source
V1 vin 0 DC 0

* Capacitors
C1 vin 2 1u
C2 2 3 1u

* Resistor
R1 2 3 1k

* Operational Amplifier
* Assume ideal op-amp using voltage controlled voltage source
Eopamp 3 0 2 0 1Meg

* Ground reference
Vgnd 2 0 DC 0

* Analysis
.op

.end