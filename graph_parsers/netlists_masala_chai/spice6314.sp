plaintext
* SPICE Netlist

V1 2 3 DC VDD

* PMOS Transistor
MP1 4 1 3 3 PMOS

* Capacitor
CL 2 3 CL_value

* Connections
* 1 -> Gate of PMOS
* 4 -> Drain of PMOS
* 3 -> Source/Body of PMOS, Ground for VDD and CL
* 2 -> Capacitor node connected to the drain