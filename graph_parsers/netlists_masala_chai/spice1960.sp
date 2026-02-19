spice
* SPICE Netlist for the given schematic

* Voltage Sources
Vpi2 3 2 DC 0

* Current Sources
Igm3 4 5 DC g_m3*(V(5) - V(4))
Igm2 0 3 DC g_m2*V(3)

* Resistors
Rpi3 Vin 4 r_pi3
Rpi2 5 3 r_pi2
RL 4 0 RL

* Nodes
Vin 0 DC 0

* End of Netlist