* NMOS Transistor
M1 2 Vin 4 4 NMOS

* Voltage Sources
Vin Vin 0 DC 0
V2 4 0 DC 2

* Resistor
R1 2 3 10k

* DC Analysis
.DC V2 0 5 0.1

* Model Definitions
.model NMOS NMOS

.end