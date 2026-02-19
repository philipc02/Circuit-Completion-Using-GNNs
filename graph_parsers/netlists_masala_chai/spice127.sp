spice
* SPICE Netlist

Q1 1 2 4 T1
Q2 3 2 6 T2
Q3 3 6 5 T3

IREF 6 5 DC 1

VCC 3 0 DC VCC
VI 2 0 DC Vi

.model T1 NPN
.model T2 NPN
.model T3 NPN

.end