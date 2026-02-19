plaintext
* SPICE Netlist for the provided schematic

Q9 9 3 4 NPN
Q10 10 4 2 NPN

I_C3 1 4 DC *Current source
I_C4 4 0 DC *Current source
I_C13 VCC 9 DC *Current source
I_C14 VCC 10 DC *Current source

R9 3 0 22k
R10 4 0 22k

VCC VCC 0 DC *Voltage supply
VB 2 0 DC *Base input

.END