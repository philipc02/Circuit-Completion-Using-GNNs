spice
* SPICE netlist for the circuit

* Voltage Source
Vx 2 0 DC 0

* Current Source
Ix 2 3 DC 0

* Resistor Ron2
Ron2 3 4 1k

* Voltage-Controlled Current Source GmVx
G1 4 0 VALUE {Gm * V(2,0)}

* Resistor Ro
Ro 4 0 1k

.END