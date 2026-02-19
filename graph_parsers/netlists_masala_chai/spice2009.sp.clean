spice
* SPICE Netlist

* Voltage Input
Vin 1 0 DC 0V

* Operational Amplifiers
* Op-Amp A1
A1 2 1 0 Vplus Vminus
.model A1 opamp

* Op-Amp A2
A2 4 3 Vcc Vee
.model A2 opamp

* Capacitors
C1 1 2 1u
C2 3 4 1u

* Resistors
Rx1 2 3 1k
Rx2 4 5 1k

* Voltage sources for op-amps
Vplus 0 Vcc DC 15V
Vminus Vee 0 DC -15V

* Load resistor
RL 5 0 1k

.END