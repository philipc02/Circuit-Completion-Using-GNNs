spice
* SPICE Netlist for the given circuit

* Resistors
Rb 3 7 Rb_value
RS 7 8 RS_value
rpi 3 4 rpi_value
RL 2 6 RL_value

* Current Source
I1 2 2 I_value
G1 5 2 (4,0) gm_value

* Voltage Nodes and Ground
V1 5 0 DC 0
V2 6 0 DC 0
V3 8 0 DC 0

* .end to signify end of the netlist
.end