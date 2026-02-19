spice
* SPICE netlist for the given schematic

* Voltage input
V1 2 7 DC 0

* Operational Amplifier
XOPAMP 2 3 3 opamp_model

* NMOS Transistor - Assume model name as nmos_model
M1 3 3 3 3 nmos_model

* Current Source
ID 3 6 DC

* Voltage Supply
VDD 4 5 DC

* Models
.model nmos_model NMOS
.model opamp_model OPAMP
.end