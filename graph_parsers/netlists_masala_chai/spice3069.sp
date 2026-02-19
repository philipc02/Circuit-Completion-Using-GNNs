* Netlist for the provided schematic
* Resistor
R1 1 2 Ro3_4/2

* NMOS Transistor M1
M1 2 3 0 0 NMOS

* NMOS Transistor M2
M2 2 3 0 0 NMOS

* Resistor Rtail
Rtail 3 4 Rtail

* Current Source gm_tail * Vout,CM
I1 4 0 tail_current

* Voltage source
v1 1 0 dc 0

.model NMOS NMOS
.dc v1 0 5 1
.end