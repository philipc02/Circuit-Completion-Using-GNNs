spice
* Netlist for the given schematic
VS 6 8 DC <value> ; Specify the DC value for Vs

* Resistors
RS 6 9 1k
RB 9 7 50k
RC 2 5 5k
RL 3 4 2k

* Dependent Current Source
G1 2 7 VALUE = {gm * V(9,7)}

* Voltage Control Node for Dependent Source
* Ensure V(9,7) refers to v_pi

* Terminator for v_o and v_ce
VCE 3 8 0 ; v_ce is output across RL

.END