* SPICE Netlist
R_GD 2 6 R_GD
R_O 6 3 R_O
R_S 2 5 R_S
I_SD 3 4 I_SD
I_SS 5 4 I_SS

* Voltage-Controlled Current Source
G1 2 4 (6 5) G

* Nodes: 
* 2 - Common source for VG and VS
* 3 - Positive terminal of ISD
* 4 - Connection to B
* 5 - Common source for VS and VBS
* 6 - Top terminal of RO and RGD
* 8 - Reference node for controlled source