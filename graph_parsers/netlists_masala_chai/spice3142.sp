spice
* Components and their nodes
M1 N4 N5 N3 N3 PMOS
I1 N4 N7 DC 1A
R1 N8 N3 1k
R2 N3 N4 1k
C1 N7 N6 1uF
Vt N2 N1 DC 5V

* Node Mapping
* N1 - Reference (Ground)
* N2 - Positive terminal of Vt
* N3 - Source of M1
* N4 - Drain of M1
* N5 - Gate of M1
* N6 - Ground node for C1
* N7 - Positive terminal of C1 (connected to current source)
* N8 - Ground node for R1

.model PMOS PMOS (VTO=-1 VMAX=100)
.END