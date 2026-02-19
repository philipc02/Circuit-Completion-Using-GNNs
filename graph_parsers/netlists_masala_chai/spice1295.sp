spice
* SPICE Netlist for the given Schematic

* Voltage Source
Vin 4 3 DC 0V

* Resistor
R1 2 3 1k  ; Assuming R1 = 1k Ohms
R2 3 0 0   ; R2 = 0 Ohms (Short circuit)

* OpAmp (ideal, requires specific model in reality)
* Connections:
* Non-inverting input: 4
* Inverting input: 2
* Output: Vout

.end