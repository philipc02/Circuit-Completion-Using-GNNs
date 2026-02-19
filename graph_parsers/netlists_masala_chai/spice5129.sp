* SPICE Netlist

* Voltage Source
V1 4 0 AC 1

* Resistors
RG 4 2 600
RF 2 3 68k
RL 2 6 10k
R2 5 2 1k
R1 3 0 20k

* NMOS Transistor
M1 3 5 0 0 NMOS

* Op-Amp
U1 2 3 2 741

* Ground
.GND 0

* End of Netlist