spice
* SPICE Netlist for the given circuit

V1 8 0 DC <Voltage_Value> ; Input voltage source
I1 6 4 <gm_ib> ; Current source with transconductance term

* Resistors
R1 5 6 <r_pi> ; r_pi between node 5 and node 6
RC 2 3 <RC_Value> ; RC between node 2 and node 3
RE 4 0 <RE_Value> ; RE between node 4 and ground
RO 3 7 <ro_Value> ; ro between node 3 and output node

* Connections
V2 7 0 DC 0 ; Output voltage measurement from node 3 to ground

.END