spice
* SPICE netlist for the schematic

Vin 1 0 DC 0

* Resistors
R1 1 2 1k
R2 2 3 1k
R3 5 0 1k
R4 3 0 1k

* Capacitors
C1 2 4 1uF
C2 3 0 1uF

* Operational Amplifier
X1 4 3 5 0 opamp

* Nodes
* 1: Vin positive terminal
* 2: Node Y
* 3: Node connected to R2, C2, and op-amp
* 4: Node X (between C1 and op-amp input)
* 5: Node Vout (op-amp output)
* 0: Ground

.end