spice
* SPICE Netlist for the given schematic

* Voltage Sources
VDD 5 0 DC <VDD_value>
Vin 2 0 DC <Vin_value> AC 1

* NMOS Transistor
M1 3 2 4 4 NMOS_MODEL

* Resistors
RF 2 3 <RF_value>
RS 4 0 <RS_value>

* Model Definition
.model NMOS_MODEL NMOS (LEVEL=1)

* End of netlist