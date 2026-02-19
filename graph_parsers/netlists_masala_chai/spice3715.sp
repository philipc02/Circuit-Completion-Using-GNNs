spice
* SPICE Netlist

V1 4 0 DC 3          ; Voltage source from node 4 to ground (3 V)
V2 0 2 DC 3          ; Voltage source from ground to node 2 (3 V)

R1 4 3 8k            ; Resistor R1 from node 4 to node 3
R2 3 2 22k           ; Resistor R2 from node 3 to node 2
RS 4 3 0.5k          ; Resistor RS from node 4 to node 3
RD 3 2 5k            ; Resistor RD from node 3 to node 2

M1 4 3 2 2 NMOS      ; NMOS: Drain at node 4, Gate at node 3, Source at node 2