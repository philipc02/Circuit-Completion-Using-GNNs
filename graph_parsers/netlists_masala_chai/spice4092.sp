spice
* SPICE Netlist for the given circuit

I1 3 4 DC <value>         ; Current source Ii from node 3 to node 4
CS 3 5 <value>            ; Capacitor CS between nodes 3 and 5
CP 4 5 <value>            ; Capacitor CP between nodes 4 and 5
Ri 6 4 <value>            ; Resistor Ri between nodes 6 and 4
Rp 2 4 <value>            ; Resistor Rp between nodes 2 and 4
Vout 5 0 DC 0            ; Voltage output at node 5

.END