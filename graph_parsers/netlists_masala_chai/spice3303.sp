I1 9 5 DC I_SS            ; Current source I_SS connected between V_DD and node 5
R1 9 6 R1                 ; Resistor R1 connected between V_DD and drain of M1
R2 6 4 R2                 ; Resistor R2 connected between drain of M1 and drain of M2
M1 6 8 5 5 PMOS           ; PMOS M1 with drain at node 6, gate at V_cont1 (node 8), source at node 5
M2 4 11 5 5 PMOS          ; PMOS M2 with drain at node 4, gate at V_cont2 (node 11), source at node 5
M3 2 2 4 4 NMOS           ; NMOS M3 with drain at node 2, gate at node 2, source at node 4
M4 2 2 10 10 NMOS         ; NMOS M4 with drain at node 2, gate at node 2, source at node 10
Vcont1 8 0 DC V_cont1     ; Control voltage source for M1
Vcont2 11 0 DC V_cont2    ; Control voltage source for M2