spice
* SPICE Netlist
* Component Definitions

Isn 5 2 DC 0 ; Current source from node 5 to node 2
D1 5 6 DiodeModel ; Diode connected between nodes 5 and 6
C1 4 6 CD_value ; Capacitor connected between nodes 4 and 6

* Model Definitions
.model DiodeModel D

* End of Netlist