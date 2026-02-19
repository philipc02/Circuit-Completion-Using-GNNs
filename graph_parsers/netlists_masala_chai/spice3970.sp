spice
* SPICE Netlist for the given schematic

Q1 2 5 3 3 2N3904

RS 7 0 10k
R1 2 5 344k
R2 5 3 467k
RE 3 33 19.6k

C2 7 2 100u
C3 2 4 100u

V+ 2 0 DC 5
V- 3 33 DC -5
V3 4 8 AC 1m

.END