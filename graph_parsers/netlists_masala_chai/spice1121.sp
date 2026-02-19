spice
* NMOS Definitions
M1 net1 Vin 0 NMOS
M2 Vout VB net1 NMOS

* Resistor
RD VDD Vout 1k

* Voltage Sources
Vin Vin 0 DC 1.8
VDD VDD 0 DC 5
VB VB 0 DC 2.5

* Model Definitions for NMOS (Generic Model)
.model NMOS NMOS (Level=1 Vto=0.7 KP=80u)

.end