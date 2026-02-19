spice
* Circuit Components
C1 1 2 C
C2 2 3 C
R1 3 4 R1
R2 2 5 R2
Vin 1 0 DC 0
XU1 5 3 4 3 opamp

* Net Definitions
* 1: Vin
* 2: Node between C1 and C2, R2
* 3: Vout, opamp output and positive input, R1 top
* 4: Ground, negative input of opamp and R1 bottom
* 5: Output of R2

* Opamp Subcircuit
.subckt opamp noninv inv out Vout
* Assuming ideal opamp
.ends opamp

.end