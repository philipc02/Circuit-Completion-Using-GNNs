spice
* SPICE netlist

* Voltage Source
Vcc 22 0 DC VCC

* Resistor
Rc 22 2 RC

* Capacitors
Cpi 2 0 Cπ
Cmu 2 22 Cμ
Ccs 2 0 CCs

* Transistor
Q1 22 2 0 Q1model

* Model Declaration (assuming a generic model for BJT)
.model Q1model NPN (IS=1E-14 BF=100)

.end