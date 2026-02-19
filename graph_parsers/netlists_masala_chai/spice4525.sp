plaintext
* Op-Amp Circuit Netlist

* Component Definitions
R1 2 2 R
R2 2 2 R
R3 2 3 R
C1 2 0 C

* Voltage Input
Vin 1 2 DC Vin

* Operational Amplifier
* Assuming ideal op-amp, the negative terminal is node 2 and the output is node 3.
.opamp U1 0 2 3 

* Voltage Output
Vout 3 0 DC 0

* End of Netlist
.end