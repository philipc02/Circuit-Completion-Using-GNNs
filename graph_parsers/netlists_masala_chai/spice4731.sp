spice
* NPN BJT Amplifier Circuit
Q1 3 2 0 NPN_BJT  ; NPN Transistor
RB 6 2 100k       ; Base resistor
RC 2 5 10k        ; Collector resistor
VBB 6 7 DC 10V    ; Base voltage supply
VCC 5 7 DC 20V    ; Collector voltage supply

.model NPN_BJT NPN (IS=1E-14 BF=50)