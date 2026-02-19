plaintext
* Netlist for the given schematic

V1 1 0 DC 10
R1 1 3 1k
D2 3 5 DModel
R2 3 4 1k
V2 4 0 DC -15
D1 3 2 DModel
I1 2 0 DC <current_value>

.model DModel D

* Node assignments
* 1: +10V
* 2: Ground
* 3: V0
* 4: -15V