spice
* SPICE Netlist for the Schematic
* NMOS: Drain Gate Source [Body]
M1 2 1 4 4 NMOS
M2 3 2 4 4 NMOS
M3 7 1 5 5 NMOS

* Current sources
I1 3 4 DC 1A
I2 6 7 DC 1A
I3 33 6 DC 1A
Iin 5 1 DC 1A

* Voltage source (VDD connected to node 33)
VDD 33 0 DC 5V

* Define model parameters for NMOS (the specifics depend on your technology)
.model NMOS NMOS(Level=1)