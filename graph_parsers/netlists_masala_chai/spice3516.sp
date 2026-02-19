spice
* SPICE Netlist

* Voltage Source
V1 3 11 DC Vi

* Resistors
RT 11 8 RT
Rin 8 9 Rin
Rb1 10 12 Rb1
Rpi1 12 2 Rpi1
Rb2 2 5 Rb2
Rpi2 5 4 Rpi2
RC 4 2 RC
RL 2 6 RL

* Capacitors
C1 2 2 C1
C2 2 4 C2

* Current Sources
Gm1 2 2 VALUE = {gm1 * V(12,2)}
Gm2 2 2 VALUE = {gm2 * V(5,2)}

* End of Netlist