spice
* SPICE netlist for the circuit

* Voltage Input
Vin 1 0 DC 0V

* Switches
S1 1 2 CK
S2 2 3 CK

* Capacitor
CH 2 0 1u

* Operational Amplifier
X1 0 2 3 OPAMP

* Control Clock
VCK CK 0 PULSE (0 5 0 1n 1n 1u 2u)

* Models
.model SW SW(Ron=1 Roff=1Meg Vt=2.5 Vh=0)
.subckt OPAMP 1 2 3
* Op-Amp model specifics
.ends OPAMP

.end