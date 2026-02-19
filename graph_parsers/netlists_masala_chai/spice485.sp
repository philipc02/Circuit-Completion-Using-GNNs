spice
* Component Declarations
V1 5 2 DC 0      ; Voltage source v1 between nodes 5 and 2
G1 6 7 5 2 gm    ; Voltage-controlled current source gm*v1 from 6 to 7, controlling voltage across 5 and 2
CPI 5 6 Cpi      ; Capacitor Cπ between nodes 5 and 6
CMU 6 5 Cmu      ; Capacitor Cµ between nodes 6 and 5
RPI 5 2 Rpi      ; Resistor rπ between nodes 5 and 2
RB 7 3 Rb        ; Resistor rb between nodes 7 and 3

* Independent Current Sources
II 2 3 Ii        ; Current source i_i from node 2 to 3
IO 4 8 Io        ; Current source i_o from node 4 to 8

.END