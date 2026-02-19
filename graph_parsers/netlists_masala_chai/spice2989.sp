plaintext
* SPICE Netlist for the given schematic

VDD 2 0 DC 5V       ; VDD is 5V DC
VF 5 0 DC 1V        ; VF is 1V DC
It 2 0 DC 1mA       ; It is a 1mA current source

RD1 3 2 10k         ; RD1 = 10k ohm
RD2 2 VDD 10k       ; RD2 = 10k ohm
R1 5 0 1k           ; R1 = 1k ohm
R2 5 0 1k           ; R2 = 1k ohm

M1 3 4 5 5 NMOS     ; NMOS transistor

.model NMOS NMOS (KP=120u W=1u L=1u)