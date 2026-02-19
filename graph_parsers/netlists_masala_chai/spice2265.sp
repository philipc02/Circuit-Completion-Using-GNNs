spice
* SPICE Netlist

* Voltage Source
VDD VDD 0 DC 1.8

* Transistors
M1 Vout Vin 0 0 NMOS_MODEL
M2 VDD Vout Vout Vout PMOS_MODEL

* Models
.model NMOS_MODEL NMOS
.model PMOS_MODEL PMOS

* End of Netlist