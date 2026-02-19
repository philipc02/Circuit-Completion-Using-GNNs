* SPICE Netlist

* NMOS transistor M1
* Drain: Node 3, Gate: Vin, Source: GND
M1 3 Vin 0 NMOS

* PMOS transistor M2
* Drain: VDD, Gate: Vb, Source: Node 3
M2 VDD Vb 3 PMOS

* Voltage source VDD
VDD VDD 0 DC 1.8V

* Define model parameters for NMOS and PMOS
.model NMOS NMOS (Level=1)
.model PMOS PMOS (Level=1)

* End of netlist