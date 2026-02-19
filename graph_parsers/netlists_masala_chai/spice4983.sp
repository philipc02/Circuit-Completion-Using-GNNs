spice
* NMOS Amplifier Circuit
V1 1 0 AC 20m
R1 5 3 1MEG
R2 4 2 2K
R3 4 7 10K
C1 4 3 <Capacitance_Value> ; Value needs definition, assumed to be required part of the circuit
M1 2 5 3 3 NMOSMODEL
* DC supply voltage
VDD 2 0 DC 15V

* .model definition for NMOS
.model NMOSMODEL NMOS (level=1 Vto=1 L=1u W=1u) ; Example NMOS model parameters

.end