spice
*MOSFET Definitions
M1 Net_2 Vin Net_2 Net_2 NMOS
M2 Net_4 Net_2 Net_2 Net_2 NMOS
M3 Net_X Net_3 Net_2 Net_2 PMOS
M4 Vout Net_3 Net_2 Net_2 PMOS

*Current Source
I1 Net_2 0 DC Iss

*Capacitors
C1 Vout Net_P C1_value
C2 Net_P 0 C2_value

*Voltage Source
VDD Net_3 0 DC VDD_value

*Simulation Commands (example)
*.OP
*.DC Vin 0 5 0.1
*.END