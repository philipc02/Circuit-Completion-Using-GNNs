* SPICE netlist for the given schematic

Vsig 4 0 DC 0 AC 1
Rsig 4 2 1k  ; Assume 1k Ohm for Rsig

Cmu1 2 3 1u  ; Assume 1uF
Cpi1 3 2 1u  ; Assume 1uF
Cn1 3 0 1u   ; Assume 1uF

Q1 3 4 0 NMOS
.model NMOS NMOS

Cs1 2 5 1u   ; Assume 1uF bypass to ground

Cmu2 2 2 1u  ; Assume 1uF
Cpi2 2 2 1u  ; Assume 1uF
Cn2 2 0 1u   ; Assume 1uF

Q2 2 2 0 PMOS
.model PMOS PMOS

C2 2 Vo 1u   ; Assume 1uF

RL Vo 6 1k   ; Assume 1k Ohm load resistor

CLplusCc2 Vo 2 1u ; Assume 1uF

Cs2 6 0 1u   ; Assume 1uF