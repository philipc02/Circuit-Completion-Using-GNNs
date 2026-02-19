plaintext
* SPICE Netlist for the given Schematic

* Voltage Sources
V1 1 5 AC 50mV
V2 2 3 AC 90mV
V3 4 5 AC 160mV
VCC 8 0 DC 18V
VEE 7 0 DC -18V

* Resistors
R1 5 2 10k
R2 5 3 20k
R3 5 6 40k
Rf 2 4 40k

* Operational Amplifier
* Node mapping: non-inverting (+), inverting (-), output
X1 6 3 4 8 7 LF157A

* Ground
.model LF157A opamp

.end