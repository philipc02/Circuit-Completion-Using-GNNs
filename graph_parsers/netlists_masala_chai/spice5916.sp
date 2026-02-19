spice
* SPICE Netlist

* Voltage Source
Vx 7 0 DC 0

* Transconductance Amplifier
G1 3 0 2 1 Gm

* Resistors
Ro 3 4 ro
RL 4 5 RL

* Current Source (inverted representation for SPICE)
I1 3 1 DC 1/Gm

* End of Netlist
.end