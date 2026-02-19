spice
* SPICE Netlist

VDD VDD 0 DC [Voltage_Value]  ; Define the supply voltage

Iin 2 0 DC [Current_Value]    ; Define the input current source

RF 4 5 [R_Value_F]            ; Feedback resistor

RD 3 VDD [R_Value_D]          ; Drain resistor

M1 3 5 2 2 NMOS_Model         ; NMOS Transistor M1

* Define NMOS model parameters
.model NMOS_Model NMOS (KP=[K_Value] VTO=[VTO_Value] ...)

* End of netlist