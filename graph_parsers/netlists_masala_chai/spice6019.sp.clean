plaintext
* SPICE netlist for given schematic

V1 2 0 DC Vdd/2 ; Voltage source at node 2
Rpi 2 6 r_pi    ; Resistor r_pi between nodes 2 and 6
G1 4 0 6 0 gm   ; CCCS: g_m * Vpi with output between nodes 4 and 0, controlling voltage across 6 and 0
Ro 4 5 r_o      ; Resistor r_o between nodes 4 and 5
Rc 4 5 RC       ; Resistor R_C between nodes 4 and 5

.END