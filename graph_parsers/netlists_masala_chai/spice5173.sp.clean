plaintext
* SPICE Netlist for the given schematic

* Voltage Sources
VCC 1 0 DC VCC
VEE 4 0 DC VEE
VIN VIN 0 DC 0

* Resistor
R1 VIN 2 R

* Diodes
D1 2 3 D
D2 2 3 D

* Op-Amp
X1 4 2 3 1 4 OPAMP

* Models
.model D D
.SUBCKT OPAMP 1 2 3 4 5
* 1 = Non-inverting input
* 2 = Inverting input
* 3 = Output
* 4 = Positive power supply
* 5 = Negative power supply
* Op-amp model components go here
.ENDS OPAMP

.END