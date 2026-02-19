* Differential Amplifier Netlist

VCC 55 0 DC <value>      ; Define VCC voltage here

* PMOS Transistors
Q3 55 7 8 PMOS           ; Q3: Drain 55, Gate 7, Source 8
Q4 55 4 7 PMOS           ; Q4: Drain 55, Gate 4, Source 7

* NMOS Transistors
Q1 4 2 3 NMOS            ; Q1: Drain 4, Gate 2, Source 3
Q2 4 2 3 NMOS            ; Q2: Drain 4, Gate 2, Source 3

* Current Source
I1 3 0 DC IEE            ; Current Source from node 3 to ground

* Nodes: 
* Node 55: VCC
* Node 7: Vb
* Node 8: Connection between Q3 source and Q1/Q2 sources
* Node 4: Vout, common drain connection for Q1 and Q2
* Node 2: Vin1, Vin2 (input signals)
* Node 3: Connection for IEE current source
* Node 0: Ground

.model PMOS PMOS (parameters) ; Define PMOS parameters
.model NMOS NMOS (parameters) ; Define NMOS parameters

.end