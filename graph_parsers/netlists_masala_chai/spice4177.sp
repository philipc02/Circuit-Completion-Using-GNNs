spice
* SPICE Netlist

V1 v1b 0 DC <value_of_v1b>
V2 v2b 0 DC <value_of_v2b>

R1 6 2 R1_value
R2 2 4 R2_value
R3 7 2 R3_value
R4 2 5 R4_value

* Operational Amplifier Model
* Input at nodes 2 (+) and 3 (-), output at node 4
X1 2 3 4 opamp

* Voltage sources connected to ground
v1b 0 v1b_value
v2b 0 v2b_value

* Resistor Values
.param R1_value = <specify_value>
.param R2_value = <specify_value>
.param R3_value = <specify_value>
.param R4_value = <specify_value>

* Example op-amp model
.subckt opamp 2 3 4
* Add op-amp details or use existing models
.ends opamp

.end