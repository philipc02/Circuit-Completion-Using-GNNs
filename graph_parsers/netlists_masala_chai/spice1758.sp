spice
* SPICE Netlist
VDD 7 0 DC 1.8V
Vin 1 0 DC 0V

* NMOS M1 (Drain, Gate, Source)
M1 3 1 0 NMOS

* PMOS M2 (Drain, Gate, Source)
M2 2 2 7 PMOS

* PMOS M3 (Drain, Gate, Source)
M3 4 2 5 PMOS

* NMOS M4 (Drain, Gate, Source)
M4 4 4 0 NMOS

* Nodes: 
* 0: GND
* 1: Vin
* 2: Common gate for M2 and input for M3
* 3: Drain of M1
* 4: Vout
* 5: Common drain for M3
* 7: VDD