spice
* SPICE Netlist

*MOSFET Definitions: Name Drain Gate Source
M3 3 6 VDD PMOS
M4 3 6 VDD PMOS

M1 3 4 5 NMOS
M2 3 4 5 NMOS

M5 5 3 2 NMOS
M6 5 3 2 NMOS

* Voltage source definitions
VDD VDD 0 DC <VDD_value>

* Ground connection
VGND 2 0 0

* Input source
Vin Vin,CM 0 DC <Vin_CM_value>

* Outputs for SPICE
VOUT Vout,C 0 

.end