plaintext
* SPICE Netlist

* PMOS Transistors: Source, Gate, Drain
M1 3 4 VDD VDD PMOS  ; Q_PA 
M2 7 3 VDD VDD PMOS  ; Q_PB
M3 3 5 5 5 PMOS      ; Q_PC 
M4 5 Y VDD VDD PMOS  ; Q_PD 

* NMOS Transistors: Drain, Gate, Source
M5 Y 2 0 0 NMOS      ; Q_NA
M6 2 6 0 0 NMOS      ; Q_NB
M7 2 1 0 0 NMOS      ; Q_NC
M8 2 5 0 0 NMOS      ; Q_ND

* Voltage supply
VDD VDD 0 DC 5V

* Inputs
VA 4 0 DC 1.8V
VB 6 0 DC 1.8V
VC 3 0 DC 1.8V
VD 5 0 DC 1.8V

* Output
.OUTPUT Y

.END