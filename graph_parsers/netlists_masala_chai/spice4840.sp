spice
* Voltage Source
Vg 11 8 DC 0

* Resistors
RG 11 2 1k
RC1 6 8 1k
RC2 4 10 1k
RL 4 3 1k

* Current Sources
I_stage1 2 8 DC 1mA
I_stage2 4 5 DC 1mA

* Ground connections
.model nmos nmos
.model pmos pmos

.end