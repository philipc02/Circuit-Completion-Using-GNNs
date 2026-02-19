plaintext
* Netlist for the given schematic

M1 2 1 0 0 NMOS   ; NMOS: Drain=2, Gate=1 (Vin), Source=0 (GND), Body=0
M2 3 2 4 4 PMOS   ; PMOS: Drain=3 (Vout), Gate=2 (Vb2), Source=4, Body=4
M3 5 6 4 4 PMOS   ; PMOS: Drain=5 (VDD), Gate=6 (Vb1), Source=4, Body=4

RD 5 3  RDvalue   ; Resistor: Between VDD (5) and Vout (3)

VDD 5 0 DC VddValue ; Voltage source: VDD
VIN 1 0 DC VinValue ; Voltage source: Vin
VB1 6 0 DC Vb1Value ; Voltage source: Vb1
VB2 2 0 DC Vb2Value ; Voltage source: Vb2

.END