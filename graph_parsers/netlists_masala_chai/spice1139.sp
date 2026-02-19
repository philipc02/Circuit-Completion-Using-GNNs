plaintext
* PMOS Transistor
M1 4 2 4 4 PMOS_MODEL

* Voltage Source 1V
V1 2 0 DC 1

* Voltage Source VDD = 1.8V
VDD 4 0 DC 1.8

* Current source Ix
Ix 2 3 DC

* Ground
Vx 3 0 DC 0

* Model for PMOS (Assuming a basic Model)
.model PMOS_MODEL PMOS (LEVEL=1 VTO=-0.7 KP=50e-6 W=1u L=1u)