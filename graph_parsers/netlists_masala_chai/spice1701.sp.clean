spice
* SPICE Netlist

* Voltage Sources
VCC 3 0 DC 15
Vin1 2 0 SIN(0 1 1k)

* Resistors
Rs 2 4 1k
Rc 3 3 1k

* Capacitors
Cμ1 3 4 10p
Cπ2 4 2 10p
CCS1 3 0 10p

* NPN Transistor
* Q1: Collector = 3, Base = 4, Emitter = 0
Q1 3 4 0 Qnlmod

.model Qnlmod NPN (IS=1.0E-14 BF=100)

* End of netlist