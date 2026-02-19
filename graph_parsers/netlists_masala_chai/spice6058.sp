spice
* SPICE Netlist for Differential Amplifier

Vsig1 1 2 DC v_sig/2
Vsig2 4 3 DC v_sig/2
VCM 6 0 DC V_CM

R1 2 2 R1Value  ; R_sig/2 (Top)
R2 3 5 R2Value  ; R_sig/2 (Bottom)

* Node mapping
* 1 - Positive Terminal of Top Vsig
* 2 - Connection point between Top Vsig and Top Resistor
* 3 - Connection point between Bottom Vsig and Bottom Resistor
* 4 - Negative Terminal of Bottom Vsig
* 5 - Output node
* 6 - Common-mode voltage source

.end