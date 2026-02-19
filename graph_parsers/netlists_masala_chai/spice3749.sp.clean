spice
* Netlist for the given schematic

V1 5 7 DC 12V
I1 2 0 DC 2mA

CC 6 7 100nF
R1 4 6 10k
R2 8 6 5k
RD 2 4 1k

MNMOS 2 3 8 8 NMOSN

* Include model definition or library for NMOS
.model NMOSN NMOS (LEVEL=1 VTO=0.7 KP=120u)

* Voltage labels
.vi 5 AC 1

.end