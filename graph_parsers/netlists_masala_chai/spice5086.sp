spice
* Op-amp Circuit Netlist

Vin 4 5 AC 1        ; AC Voltage Source
R1 3 6 1k           ; Resistor R1 connected between node 3 and ground
Rf 3 2 10k          ; Resistor Rf connected between node 3 and node 2
XU1 2 3 Vout opamp  ; Op-amp with non-inverting input at node 3, inverting input at node 2, and output at Vout
* Note: opamp is a placeholder for the actual op-amp model

* Ground nodes
0 5
0 6

.end