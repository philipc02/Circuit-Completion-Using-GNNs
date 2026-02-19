plaintext
* Spice Netlist
R1 1 2 51k
R2 2 4 51k
C1 2 3 330p
C2 4 7 440p
XOPAMP 6 7 5 opamp

* Voltage source at input
Vin 1 0 DC 0

* Subcircuit model for the operational amplifier
.subckt opamp in+ in- out
* Idealized opamp model
* (modeling details depend on specific opamp characteristics)
.ends opamp

.END