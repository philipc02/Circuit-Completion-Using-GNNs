spice
* SPICE netlist for the given schematic

* Voltage Sources
V1 9 7 DC 0
V2 2 8 DC 0

* Capacitors
C1 7 2 Cs
C2 5 6 Cl
C3 2 3 Cf

* Operational Amplifier
* Assume ideal op-amp model
XOPAMP 2 3 4 OPAMP_MODEL

* Connections
* Connecting node 4 to Vout
Vout 4 0 DC 0

* Model Definitions
.model OPAMP_MODEL opamp