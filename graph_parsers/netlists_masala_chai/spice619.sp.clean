* SPICE Netlist for the provided schematic

V1 3 0 Vin       ; Voltage source Vin connected from net 3 to ground
R1 3 6 1k        ; Resistor R1 connected between net 3 and net 6
R2 6 3 1k        ; Resistor R2 connected between net 6 and ground (net 3)
D1 6 5 Dmodel    ; Diode D1 connected between net 6 and net 5, cathode side to net 5
Rload 5 0 1k     ; Load resistor (implicit) connected between net 5 and ground

.MODEL Dmodel D  ; Diode model definition
.END