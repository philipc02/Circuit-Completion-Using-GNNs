plaintext
* SPICE netlist

M2 3 Vin 4 4 NMOS    ; NMOS M2: drain=3, gate=Vin, source=4, body=4
M1 2 4 4 4 NMOS      ; NMOS M1: drain=2, gate=4, source=4, body=4

RD 3 2 RD_Value      ; Resistor RD: node1=3, node2=2
RS 4 0 RS_Value      ; Resistor RS: node1=4, node2=0

Vin Vin 0 DC Vin_Value ; Voltage source Vin: positive=Vin, negative=0
V1 0 4 DC V1_Value     ; Voltage source V1: positive=0, negative=4
VDD 3 0 DC VDD_Value   ; Voltage source VDD: positive=3, negative=0

* .model definitions here
.model NMOS NMOS(...) ; Add NMOS model parameters here

.end