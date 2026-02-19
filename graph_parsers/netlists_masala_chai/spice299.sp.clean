spice
* Spice Netlist for the Given Schematic

V1 5 0 DC 0

Rs 5 6 1k  ; Assuming Rs = 1kOhm (value can be adjusted)
RL 3 7 1k  ; Assuming RL = 1kOhm (value can be adjusted)

M1 3 6 7 7 NMOS_MODEL  ; NMOS with drain=3, gate=6, source=7, body=7

Vout 3 0  ; Output voltage VO across node 3 and ground

.model NMOS_MODEL NMOS (LEVEL=1)  ; NMOS device model (adjust parameters as needed)

.end