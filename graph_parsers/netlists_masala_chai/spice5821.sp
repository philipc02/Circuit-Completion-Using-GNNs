plaintext
* SPICE Netlist for Given Schematic

V1 5 6 DC <DC_Voltage_Value> ; Voltage source (vi)
Q1 4 5 2 NPN                 ; NPN BJT Transistor
RL 2 3 <R_L_Value>           ; Load Resistor

* Ground Node
3 0                          ; Connect node 3 to ground
6 0                          ; Connect node 6 to ground

.END