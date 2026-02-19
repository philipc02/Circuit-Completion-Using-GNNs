plaintext
* SPICE Netlist

VDD 7 0 DC 5V  ; Voltage source VDD, connected to node 7 and ground

M3 3 3 6 6 NMOS ; NMOS transistor M3: Drain=3, Gate=3, Source=6, Body=6
M5 4 3 2 2 PMOS ; PMOS transistor M5: Drain=4, Gate=3, Source=2, Body=2

I1 2 1 DC 1mA  ; Current source I1, connected between node 2 and ground

Cc 3 2 10pF    ; Capacitor Cc, connected between node 3 and node 2

ISS 3 7 DC 1mA ; Current source ISS, connected between node 3 and node 7

.end