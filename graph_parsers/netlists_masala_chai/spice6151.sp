plaintext
* SPICE Netlist for the given schematic

Vsig 9 0 DC 0 AC 1 SIN(0 1m 1k)
R1 8 5 100k
RG 5 7 10Meg
R2 3 0 6.8k
R3 7 2 3k
R4 2 6 1k
C1 8 5 0.1u
C2 2 6 1u

* NPN BJT, Q1
Q1 5 5 3 NPN

* PNP BJT, Q2
Q2 2 2 4 PNP

* Voltage source
Vcc 7 0 DC 5

.model NPN NPN(is=1e-14 bf=100)
.model PNP PNP(is=1e-14 bf=100)

.end