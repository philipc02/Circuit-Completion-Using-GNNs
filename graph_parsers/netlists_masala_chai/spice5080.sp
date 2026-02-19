spice
* Summing Amplifier Circuit
R1 5 3 1k
R2 2 3 1k
R3 2 2 1k
Rf 3 4 1k

* Voltage Inputs
V1 5 0 DC 0
V2 2 0 DC 0
V3 2 0 DC 0

* Operational Amplifier
* Assuming ideal op-amp, using SPICE element 'E' for dependent source
E1 3 0 3 0 999k

* Ground
V0 0 7 DC 0

.end