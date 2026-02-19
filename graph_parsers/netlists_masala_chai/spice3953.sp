spice
* SPICE netlist for the given schematic

V1 5 0 AC 12m
V+ 2 0 DC 5
V- 6 0 DC -5

RC 1 2 10k
RB 5 3 100k
RE1 3 4 240
RE2 4 6 20k

C1 5 3 100u
C5 4 6 100u

Q1 1 3 4 2N3904

.END