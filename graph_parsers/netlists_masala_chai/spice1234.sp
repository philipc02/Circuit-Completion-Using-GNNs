spice
* NMOS and PMOS models
.model NMOS NMOS
.model PMOS PMOS

* Voltage sources
VDD VDD 0 DC 5

* Transistors
M2 Vout Vin VDD VDD PMOS
M1 4 Vb 0 0 NMOS
M3 4 4 0 0 NMOS

* Define nodes
.node Vin Vb Vout VDD 0

.end