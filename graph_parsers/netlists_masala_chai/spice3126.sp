spice
* MOSFETs and Current Sources
M1 3 2 0 0 NMOS  ; M1 (drain = X, gate = Vin, source= 0 , body = 0)
M2 5 2 0 0 NMOS  ; M2 (drain = Y, gate = Vin, source= 0 , body = 0)
M3 3 1 4 4 PMOS  ; M3 (drain = X, gate = Vb1, source= VDD, body = VDD)
M4 5 1 4 4 PMOS  ; M4 (drain = Y, gate = Vb1, source= VDD, body = VDD)
M5 1 1 4 4 PMOS  ; M5 (drain = Vout1, gate = Vb1, source= VDD, body = VDD)
M6 5 1 4 4 PMOS  ; M6 (drain = Vout2, gate = Vb1, source= VDD, body = VDD)
M7 0 2 0 0 NMOS  ; M7 (drain = Vout1, gate = Vb1, source= 0 , body = 0)
M8 4 2 0 0 NMOS  ; M8 (drain = Vout2, gate = Vb1, source= 0 , body = 0)

* Current Source
Iss 0 3 DC  ; Iss connected between source of M1, M2 to ground

* Voltage Definitions
VDD 4 0 DC ; DC voltage source for VDD
Vb1 2 0 DC ; DC voltage source for Vb1
Vin 2 0 DC ; DC input for Vin

* Outputs
Vout1 1 0   ; Output Vout1
Vout2 5 0   ; Output Vout2