spice
* SPICE Netlist for the given CMOS circuit

VDD VDD 0 DC 5V

* N-channel MOSFETs
MN1 2 B 3 NMOS
MN2 Y A 4 NMOS

* P-channel MOSFETs
MP1 2 C VDD PMOS
MP2 3 D VDD PMOS

* Nodes:
* 2 : Common drain for MP1 and MN1
* 3 : Common drain for MP2 and MN1 source
* 4 : Drain of MN2 and source of MN2
* Y : Output node

.model NMOS NMOS
.model PMOS PMOS