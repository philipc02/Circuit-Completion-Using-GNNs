spice
* SPICE Netlist for the given schematic
* Voltage Source
Vsig 6 0 DC 0

* Resistors
R1 6 4 10k
R2 4 7 100k
R3 8 0 1k
R4 9 0 1k

* Capacitors
C1 7 2 inf
C2 8 9 inf

* Transistor
Q1 5 2 8 BJT_MODEL ; Assuming a generic BJT model is declared elsewhere

* Power Supply
V1 5 0 DC 3

* Model declaration (Generic BJT)
.model BJT_MODEL NPN ; Replace NPN with PNP if necessary

* Node Voltage
Vout 2 0