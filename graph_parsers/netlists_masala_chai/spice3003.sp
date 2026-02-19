* SPICE Netlist

* Node Mapping
* 1 -> Ground
* 2 -> Source (Drain of M1 and M3, one terminal of RF)
* 3 -> Gate of M3
* 4 -> Drain of M2, Source of M3 (Connected to VDD)
* 5 -> Source of M2
* 6 -> Drain of M1, RF connection node
* 7 -> VDD
* 8 -> Gate of M2
* 9 -> Gate of M1

* NMOS Transistors: (Name Drain Gate Source)
M1 6 9 2 NCH
M3 2 3 4 NCH

* PMOS Transistor: (Name Drain Gate Source)
M2 4 8 5 PCH

* Resistors
RS1 2 1 100K
RF 6 2 10K
RD2 2 1 10K

* Voltage Sources
VDD 7 1 DC 5V
Vb 8 1 DC 1V
Vin 9 1 DC 1V

.end