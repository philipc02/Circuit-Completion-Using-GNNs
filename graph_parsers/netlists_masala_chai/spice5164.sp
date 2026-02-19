spice
* SPICE Netlist for the Schematic
V1 4 5 DC Vin        ; Voltage source
R1 3 4 R1            ; Resistor R1 in parallel with R3
C1 3 2 C1            ; First capacitor
C2 2 0 C2            ; Second capacitor to ground
R2 6 2 R2            ; Resistor R2
XOPAMP 0 2 Vout OPAMP ; Op-amp with inverting input at node 2, non-inverting input grounded

* Model for the op-amp
.model OPAMP opamp
.ends

* End of netlist