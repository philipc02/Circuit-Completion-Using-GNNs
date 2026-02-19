spice
* SPICE netlist for the given schematic

* Voltage-controlled current source
G1 6 5 5 0 gm

* Resistor
R1 6 7 RL

* Capacitor
C1 4 2 Ctotal

* Node Definitions
* 6 - Node D (Drain)
* 5 - Node connected to current source
* 2 - Ground
* 7 - Node connected at the bottom of RL
* 4 - Node Vo (Output)

.end