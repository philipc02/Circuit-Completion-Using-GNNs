plaintext
* SPICE netlist for given schematic

V1 3 0 DC 0 AC 50m
VDD 2 0 DC 9

* MOSFET
M1 2 3 0 0 2N7000 

* Resistors
R1 2 4 2Meg
R2 3 4 1Meg
RD 2 5 150
RL 5 0 1k

* Capacitors
C1 7 3 Cvalue ; Specify Cvalue for input coupling capacitor
C2 6 5 Cvalue ; Specify Cvalue for output coupling capacitor

* Voltage Source
V_supply 2 0 DC 9

.tran 1n 10n 
.end