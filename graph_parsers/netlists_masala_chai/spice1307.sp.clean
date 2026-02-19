spice
* SPICE Netlist for the given schematic

* Voltage Source
Vin 5 0 DC

* Resistor
R1 5 1 1k

* MOSFET (Assuming NMOS based on source arrow direction)
M1 3 1 2 2 NMOS_MODEL

* Operational Amplifier
* (Assuming ideal or predefined subcircuit)
.subckt OPAMP +in -in out
* (Ideal op-amp model goes here)
.ends

XU1 0 2 1 OPAMP

* Analysis
.dc Vin 0 5 0.1

* Model parameters for an NMOS
.model NMOS_MODEL NMOS (Level=1)

.end