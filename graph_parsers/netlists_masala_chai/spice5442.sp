spice
* SPICE Netlist
V1 9 2 AC 1 SIN(0 1 15.9k)    ; Voltage source with AC frequency 15.9 kHz
R1 5 9 5.1k                  ; Resistor R1 = 5.1kΩ
RW 8 3 25                    ; Resistor RW = 25Ω
L1 3 6 5m                    ; Inductor L = 5.0mH
C1 6 2 0.022u                ; Capacitor C = 0.022µF