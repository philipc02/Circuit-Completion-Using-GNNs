plaintext
* SPICE Netlist
V1 4 0 DC 15V   ; Voltage Source V_CC = 15V
R1 4 2 1Meg     ; Resistor 1MΩ
D1 2 0 D        ; Diode connected to node 2
Q1 3 2 5 NPN    ; NPN Transistor, C=3, B=2, E=5

.model NPN NPN(IS=1E-14 BF=100)
.model D D(IS=1E-14)