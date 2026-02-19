plaintext
* SPICE Netlist
* Voltage Sources
Vout 3 0 DC 0
Vin 4 0 DC 0

* Operational Amplifier
XU1 2 4 4 OPAMP

* Diode
D1 2 4 DiodeModel

* Resistors
RP 2 3 1k
R1 3 0 1k

* Models
.model DiodeModel D
.model OPAMP opamp