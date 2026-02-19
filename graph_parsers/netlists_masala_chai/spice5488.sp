spice
* NMOS Amplifier Circuit

V1 7 8 DC 0       ; Input voltage source Vin
VDD 9 8 DC VDD    ; DC supply voltage VDD

* Capacitors
C1 7 4 C1         ; Capacitor C1
C2 9 2 C2         ; Capacitor C2
C3 2 3 C3         ; Capacitor C3

* Resistors
R1 4 6 R1         ; Resistor R1
R2 4 8 R2         ; Resistor R2
RD 9 2 RD         ; Resistor RD
RS 2 3 RS         ; Resistor RS

* Transistor
M1 9 4 2 2 NMOS   ; NMOS Transistor M1 (D G S B)

* Simulation Commands
.end