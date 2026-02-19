plaintext
* Circuit Netlist

Vin 5 0 DC 0          ; Input voltage source connected to net 5
Rin 5 0 1k            ; Rin connected between net 5 and ground
E1 2 0 3 5 1          ; Voltage controlled voltage source with gain A0 connected between nodes 2-0 with control input 3-5
Rout 2 4 1k           ; Rout connected between net 2 and net 4
Vout 4 0 DC 0         ; Output voltage