spice
* SPICE Netlist for the given schematic

VCC 5 8 DC 12V

Q1 2 3 7 NPN
Q2 5 2 8 NPN

R1 5 2 1k
R2 2 9 1k
R3 3 7 1k

Cj 0 1 10u
Co 4 10 10u
CE 7 8 10u

Vinput 0 1 AC 1V

* End of netlist