spice
* SPICE Netlist
* Components
R1 1 2 47k
R2 2 5 47k
R3 5 6 30k
R4 5 3 51k
C1 5 6 330p
C2 2 3 330p

* Op-Amp
* Ideal Op-Amp model assuming large gain
* and Ground connected to node 3
XOPAMP 3 5 6 opamp

* Voltage source for input
Vin 1 0 DC 0

* Include op-amp model
.include opamp.lib

.end