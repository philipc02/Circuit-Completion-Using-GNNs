* BJT Parameters (assuming NPN and PNP where needed)
.model NPN Q(IS=1E-14 BF=200)
.model PNP Q(IS=1E-14 BF=200)

* Transistors
Q1 2 1 6 NPN
Q2 3 2 6 PNP
Q3 4 3 6 NPN
Q4 5 4 6 PNP

* Current Source
I1 7 2 DC IQ

* Resistors
R1 3 4 R1
R2 4 5 R2
Rx 8 5 Rx

* Voltage Source for Offset Nulling (assuming a ground reference at node 8)
V1 8 1 V+

* Connections
V+ 7 0 DC V+