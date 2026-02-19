spice
* SPICE Netlist for the given schematic

* Voltage Source
V1 5 0 DC Vin

* NMOS Transistor
M1 2 4 3 3 NMOS

* Resistors
R1 22 2 R1_value
R2 4 0 R2_value
RD 2 3 RD_value
RS 3 0 RS_value

* Capacitors
Ci 4 2 Ci_value
Cb 3 0 Cb_value

* Power Supply
VDD 22 0 DC VDD_value

* .model and .options are required for simulating the circuit
.model NMOS NMOS_level_params
.options reltol=1e-3
.end