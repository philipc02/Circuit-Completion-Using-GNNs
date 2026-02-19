spice
* NMOS Amplifier Circuit

Vin in 0 DC 0
VDD 2 0 DC 1.8

* NMOS transistor
* M1 <drain> <gate> <source> <bulk> <model>
M1 2 Vin 3 3 NMOS_MODEL

* Resistors
RD 2 3 RD_VALUE
RS 3 0 RS_VALUE

* Models
.model NMOS_MODEL NMOS(Level=1)

* End of netlist