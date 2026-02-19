* SPICE Netlist for the given schematic

Vgs 1 7 DC 0
Igm 1 3 DC gm_vgs
Igmb 3 2 DC gmb_vbs
Ro 3 4 ro
Vt 6 3 DC 0
It 6 7
Rs 7 7 9 RS

* Voltage sources
Vgs 1 7 DC

* Current sources
Igm 1 3 DC gm_vgs
Igmb 3 2 DC gmb_vbs

* Resistors
Ro 4 6 ro
Rs 9 7 RS

* Control source
Vt 6 7 DC

* End of netlist
.end