spice
* SPICE netlist for differential amplifier circuit

* Voltage Sources
V1 3 2 Vsd
V2 2 3 Vsd
VCM1 2 4 VCM
VCM2 2 2 VCM

* Resistors
RA1 4 1 RA
RB1 1 5 RB
RA2 2 2 RA
RB2 5 2 RB

* Operational Amplifier (Ideal)
* Assuming ideal op-amp behavior with high gain
EOPAMP 5 2 3 2 100k

.END