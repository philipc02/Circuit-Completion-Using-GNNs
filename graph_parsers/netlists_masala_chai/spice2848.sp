spice
* SPICE Netlist for the given circuit

* Voltage Source
Vin 2 0 DC Vin

* PMOS Transistor
M2 4 2 7 7 PMOS_Model

* NMOS Transistor
M1 4 2 6 6 NMOS_Model

* Power Supply
VDD 7 0 DC VDD

* Models (Assuming they are defined elsewhere)
.model PMOS_Model PMOS
.model NMOS_Model NMOS

* End of netlist