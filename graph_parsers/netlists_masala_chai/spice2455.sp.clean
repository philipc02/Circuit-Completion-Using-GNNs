plaintext
* SPICE Netlist

Vg 3 0 DC <value>          ; VG voltage source, define the DC value
Vx 2 0 DC <value>          ; VX voltage source, define the DC value
Ix 5 2 DC <value>          ; IX current source, define the DC value

* PMOS Transistor
M1 5 3 4 4 PMOS_MODEL      ; PMOS: Drain=5, Gate=3, Source=4, Body=4

* NMOS Transistor
M2 6 3 8 8 NMOS_MODEL      ; NMOS: Drain=6, Gate=3, Source=8, Body=8

.model PMOS_MODEL PMOS (kp=..., vt0=..., lambda=...)  ; Define PMOS model parameters
.model NMOS_MODEL NMOS (kp=..., vt0=..., lambda=...)  ; Define NMOS model parameters

.end