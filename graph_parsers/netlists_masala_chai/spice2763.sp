spice
* SPICE Netlist

* NMOS Transistor Definition
M1 3 6 1 1 NMOS_MODEL

* Voltage Source
Vx 2 1 DC 0

* Current Source
Ix 6 2 DC 0

* Resistor
RD VDD 3 1k

* Capacitors
CGD 6 3 1p
CDB 3 4 1p

* DC Supply Voltage
VDD VDD 0 DC 5V

* Model Definition (Placeholder)
.model NMOS_MODEL NMOS