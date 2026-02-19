spice
* SPICE netlist for the given schematic

* Voltage sources
Vsig 1 7 DC 0
Vplus 5 0 DC 5
Vminus 3 0 DC -10

* Current sources
I1 5 2 DC 1m
I2 6 2 DC 0.5m

* Transistors
Q1 2 7 6 NPN
Q2 2 3 4 NPN

* Resistors
R1 7 2 10k
R2 2 3 10k

* Capacitor (Assumed from the symbol)
C1 2 0 1u

* End of netlist