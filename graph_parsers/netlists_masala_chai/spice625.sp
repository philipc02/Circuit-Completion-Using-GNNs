spice
* SPICE Netlist for the Given Schematic

* Resistors
R1 4 5 1k
R2 2 2 1k

* Current Sources
I_R2 4 2 DC 1e-9  ; Current source approximation for ∞ current
I1 3 3 DC 1e-9    ; Current source approximation for ∞ current

* Voltage Sources
V_Don1 5 3 DC 0.7 ; Voltage drop between nodes (assumed typical diode drop)
Vin 5 0 DC 5      ; Input voltage

* Control Section (assuming input and output as nodes)
.control
run
.endc

.end