* Differential Amplifier SPICE Netlist
VCC 4 0 DC VCC
VID 11 0 DC VID
VEE 9 0 DC -VEE

I1 10 4 DC 2I1

* Resistors
RE1 6 2 RE
RE2 6 2 RE

* Transistors
Q1 8 11 2 NPN
Q2 6 5 2 NPN
Q3 8 3 9 PNP
Q4 5 7 9 PNP

.model NPN NPN
.model PNP PNP

.end