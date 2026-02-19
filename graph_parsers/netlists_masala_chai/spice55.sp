* SPICE netlist for the given schematic

* Voltage source
Vt 1 0

* Current sources
It 7 10 DC
Ib 4 2 DC

* Resistors
Rpi 3 4 rpi_value
Ro 2 12 ro_value
RL 12 6 RL_value

* Connections
* Vt is connected between node 1 and ground (node 0)
* It is connected between node 7 and 10
* Rpi is connected between nodes 3 and 4
* Ib (beta*Ib current source) is connected between nodes 4 and 2
* Ro is connected between nodes 2 and 12
* RL is connected between nodes 12 and 6

.end