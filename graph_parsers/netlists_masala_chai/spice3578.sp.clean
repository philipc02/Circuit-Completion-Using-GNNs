spice
* SPICE Netlist for NPN Transistor Circuit

* Voltage Source
V1 3 6 DC 6V   * V_C = 6 V

* Resistors
R1 2 3 180k    * 180 kΩ resistor
R2 8 2 6k      * 6 kΩ resistor
R3 4 7 5k      * 5 kΩ resistor

* NPN Transistor
Q1 4 2 7 QNL   * NPN Transistor Q1

* Model for NPN Transistor
.model QNL NPN (Is=1e-15 Bf=100)

* Voltage source for input
Vin 8 6 DC 0V  * V_IN, connect to node 8

* Ground
R4 7 6 0       * Ground reference

.end