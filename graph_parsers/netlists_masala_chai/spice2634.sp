* SPICE Netlist

* Voltage Source
VDD 5 0 DC VDD_value

* Resistors
RD1 5 2 RD_value
RD2 5 3 RD_value
RP 2 7 RP_value

* PMOS Transistors
M1 2 7 5 5 PMOS_model
M2 3 8 5 5 PMOS_model

* NMOS Transistor
M3 2 4 1 1 NMOS_model

* Technical Information
.model PMOS_model PMOS (kp=1u Vto=-1)
.model NMOS_model NMOS (kp=1u Vto=1)

* Nodes Mapping
* 1: Ground
* 2: Vout
* 3: Connection between RD and M2
* 4: Vb
* 5: VDD
* 6: Connection between RD2 and M2
* 7: Vin1
* 8: Vin2

.end