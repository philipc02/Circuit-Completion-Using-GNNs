* SPICE Netlist
I1 6 0 DC 0  ; Current source I_in
C1 2 7 CP    ; Capacitor C_P
L1 3 5 LP    ; Inductor L_P
R1 4 44 RP   ; Resistor R_P

* Additional connections based on the schematic
* Node 6 is common for I_in, C_P, L_P, and R_P
Vout 4 0     ; Output voltage node