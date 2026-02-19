plaintext
* SPICE Netlist for the provided circuit

VCC 2 0 DC 15V
VEE 4 0 DC -15V
Vb1 26 0 DC 5V
Vin 3 0 AC 1

* NPN Transistors
Q1 2 5 3 NPN
Q2 5 3 4 NPN
Q3 26 2 5 NPN
Q4 3 3 4 NPN

* Diodes
D1 2 3 DModel
D2 3 4 DModel

* Load Resistor
RL 5 0 1k

.model NPN NPN   (IS=1E-14 BF=100)
.model DModel D  (IS=1E-14)

.end