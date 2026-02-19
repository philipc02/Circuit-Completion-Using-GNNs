spice
* Netlist for the given circuit

* Current Source
I1 6 2 DC Ib

* Resistors
Rpi 6 7 rpi
Ro 4 2 ro

* Capacitors
Cmu 6 4 Cmu
Cpi 6 7 Cpi

* Voltage-Controlled Current Source
Gm 4 2 6 7 gm

* Voltage Source for Vpi
Vpi 6 7 DC Vpi

* Other nodes
* Node 7 is connected to ground
Vgnd 7 0 0

.end