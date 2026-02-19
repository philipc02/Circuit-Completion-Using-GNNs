spice
* SPICE Netlist

* Voltage Source
Vin 1 0 DC 0

* Resistors
R1 1 3 R1_value
R2 3 7 R2_value
R3 6 5 R3_value
R4 5 4 R4_value

* Capacitors
C1 3 5 C_value
C2 2 4 C_value

* Operational Amplifiers
* Op-amp 1
XU1 0 3 3 2 opamp

* Op-amp 2
XU2 2 5 5 2 opamp

* Op-amp 3
XU3 2 4 4 6 opamp

* Model Definitions
.subckt opamp noninv inv out vcc
* Model details would be defined here
.ends opamp