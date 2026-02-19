spice
* SPICE Netlist for the given schematic

* Voltage Source
VDD VDD 0 DC <value> ; Replace <value> with actual voltage

* NMOS Transistor
M1 Vout X RF RF NMOS ; Replace 'NMOS' with model name

* Resistors
Rs Vin X <value> ; Replace <value> with actual resistance of Rs
RF RF 0 <value> ; Replace <value> with actual resistance of RF
RL Vout 0 <value> ; Replace <value> with actual resistance of RL

* Input Voltage Source
Vin Vin 0 DC 0V ; or AC 0 1 SIN(0 1 1k) for example

* .MODEL statement for NMOS (example model)
.model NMOS NMOS (LEVEL=1 VTO=1 KP=100u)

.end