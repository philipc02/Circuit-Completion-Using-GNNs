spice
* SPICE Netlist for the given circuit

Vgs 1 2 DC <value_of_Vgs>
Vi 1 4 DC <value_of_Vi>

Gm 4 2 POLY(1) Vgs 0 <value_of_gm>

Ron 4 5 <value_of_ron>
Rop 5 3 <value_of_rop>

* Define ground
0 4 0

* Output voltage
Vout 5 3

.end