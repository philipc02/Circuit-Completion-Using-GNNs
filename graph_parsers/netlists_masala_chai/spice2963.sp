spice
* NMOS and PMOS definitions
.model NMOS NMOS
.model PMOS PMOS

* Voltage Source
VDD 23 0 DC VDD_value

* Transistors
M1 3 7 6 6 NMOS
M2 2 4 23 23 PMOS

* Resistors
RS1 3 0 RS_value
RF1 6 0 RF_value
RS2 5 0 RS_value
RF2 5 0 RF_value
RD2 2 0 RD2_value

* Node connections
* X: between M1 (drain) and M2 (gate)
* Y: between M2 (source) and RD2, connecting to node 4