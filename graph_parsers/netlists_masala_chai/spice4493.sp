spice
* Operational Amplifier Circuit

* Voltage Input
Vin vi 0 DC 0

* Resistors
R1 vi 2 10k
R2 2 3 30k
R3 3 4 10k
R4 4 0 10k
R5 5 3 20k

* Op-Amps
* First Op-Amp
XU1 0 2 3 OPAMP
* Second Op-Amp
XU2 0 4 5 OPAMP

* Analysis
.TRAN 1u 10m
.END