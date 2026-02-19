spice
** SPICE Netlist **

* Transistors
Q1 6 7 4 NPN
Q2 6 3 2 NPN
Q3 5 8 5 PNP
Q4 8 6 9 NPN

* Resistors
R_R 6 7 ValueR
R_RC 6 3 ValueRc

* Power Supplies
VCC 6 0 DC VCCValue
VEE 5 0 DC VEEValue

* Output
Vout 3 0

.model NPN NPN
.model PNP PNP