* SPICE Netlist

* Voltage Source
VDD 3 0 DC 5V

* Current Source
Iin 2 0 DC 10uA

* PMOS Transistor
* M1: Drain(Gate) - Source - Body
M1 3 4 3 PMOS_MODEL

* NMOS Transistor
* M2: Drain(Source) - Gate - Source
M2 4 2 0 NMOS_MODEL

* Capacitors
C1 3 Vout 1uF
C2 4 0 1uF

* Resistor
RD 3 Vout 1k

* Models
.model PMOS_MODEL PMOS (LEVEL=1 VTO=-1)
.model NMOS_MODEL NMOS (LEVEL=1 VTO=1)

.end