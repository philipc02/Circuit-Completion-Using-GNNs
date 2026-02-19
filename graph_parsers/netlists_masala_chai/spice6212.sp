plaintext
* SPICE Netlist for the given circuit

* NMOS Transistor
M1 1 3 5 5 NMOS

* PMOS Transistor
M2 5 2 6 6 PMOS

* Resistor
RL 1 4 1k

* Voltage Sources
VCC1 7 0 DC VCC
VCC2 0 6 DC VCC

* Input Voltage
Vin 8 0 DC vi

* Connections
V1 8 3 DC 0
V2 0 2 DC 0

* Default Models for Transistors
.model NMOS NMOS (Level=1)
.model PMOS PMOS (Level=1)

.end