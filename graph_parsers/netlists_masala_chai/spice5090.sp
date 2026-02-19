spice
* Operational Amplifier Circuit Netlist

* Voltage Source
Vin 5 0 AC 1

* Resistors
R1 5 2 15k
Rf 3 2 300k

* Operational Amplifier
* Connections: (non-inverting input, inverting input, output)
X1 0 2 3 opamp

* Power Supplies for the Op-Amp
VCC+ 4 0 DC 15V
VCC- 2 0 DC -15V

* Output
Rload 3 0 10k ; Load resistor if needed

* Model for LF157A
.model opamp op(GBW=20Meg AV=100k)

.end