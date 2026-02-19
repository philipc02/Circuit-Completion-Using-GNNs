spice
* SPICE Netlist

V1 6 2 DC 15V          ; Voltage source V+ = +15V
RC 3 6 1k              ; Resistor RC between nets 3 and 6
RE 2 3 1k              ; Resistor RE between nets 2 and 3
R1 4 5 1k              ; Resistor between nets 4 and 5

Q1 3 4 2 NPN           ; NPN Transistor, Collector=3, Base=4, Emitter=2
D1 2 5 D               ; Diode D, Anode=2, Cathode=5

.model D D             ; Diode model
.model NPN NPN         ; Transistor model

.end