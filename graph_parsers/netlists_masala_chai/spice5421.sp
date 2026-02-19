spice
* SPICE Netlist
Vs 8 2 AC 10 SIN(0 10 250k)  ; Voltage source with 10V amplitude and 250 kHz frequency
R1 3 2 4.7k                  ; Resistor R1 between nodes 3 and 2
L1 2 6 5m                    ; Inductor L1 between nodes 2 and 6
R2 6 4 3.3k                  ; Resistor R2 between nodes 6 and 4
L2 4 2 2m                    ; Inductor L2 between nodes 4 and 2