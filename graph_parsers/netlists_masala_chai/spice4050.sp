spice
* SPICE Netlist for the Circuit

Vi 6 0 DC 0
RS 6 4 1k
rpi 4 0 10k
Igm 3 4 Vpi 1m
RE 2 3 500
CE 2 3 10u
RC 4 5 500
Vpi 0 4 DC 0

.tran 1n 10u
.end