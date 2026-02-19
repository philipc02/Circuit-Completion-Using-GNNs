spice
* SPICE Netlist for the Circuit

C1 Vin X 1u

M1 X X 0 0 NMOS

V1 Vin 0

* Connect Vout to node X
Vout X 0

.model NMOS NMOS

.end