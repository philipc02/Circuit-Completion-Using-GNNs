spice
* NMOS Transistor Amplifier Circuit
M1 3 5 6 6 NMOS_MODEL

* Resistors
RG1 2 VDD R_G1
RG2 2 0 R_G2
RD 3 VDD R_D
RS 6 0 R_S

* Voltage Source
VDD VDD 0 DC <Value> ; Replace <Value> with actual DC voltage value

* .model declaration for NMOS
.model NMOS_MODEL NMOS (LEVEL=1 TOX=<Value> NSUB=<Value> VTO=<Value> KP=<Value>)