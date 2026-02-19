spice
* SPICE Netlist for the Given Circuit

Iin 3 4 DC 0.1A  ; Current source with 0.1 A for example

D1 2 4 Dmodel    ; Diode D1 connected from node 2 to node 4
D2 3 2 Dmodel    ; Diode D2 connected from node 3 to node 2

R1 2 4 1k        ; Resistor R1 connected from node 2 to node 4 with 1k ohm

.model Dmodel D  ; Diode model Dmodel

.end