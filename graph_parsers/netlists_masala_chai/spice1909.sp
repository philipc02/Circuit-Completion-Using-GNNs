* Component List
* Capacitors
C1 2 X 10nF
C2 X Y 10nF
C3 Y 3 10nF

* Resistors
R1 X 0 10k
R2 Y 0 10k
R3 3 0 10k

* Voltage Controlled Voltage Source (Operational Amplifier)
E1 4 0 3 2 -A0

* Nodes
* 2: Input
* 3: Connection between C3, R3, and non-inverting terminal of Op-Amp
* 4: Output (Vout)

* Simulation Control
.tran 1us 10ms
.end