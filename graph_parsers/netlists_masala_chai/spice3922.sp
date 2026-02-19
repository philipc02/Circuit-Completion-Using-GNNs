plaintext
* SPICE Netlist for the given schematic

V1 10 0 DC 10V
V2 6 0 DC -10V
Vs 1 10 DC 0V

R1 1 2 100k
R2 2 5 40k
RC1 8 4 3k
RE1 5 0 1k
RE2 3 6 5k

CC 2 5
Q1 4 2 5 NPN
Q2 3 4 6 NPN

* End of netlist