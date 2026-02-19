spice
* SPICE Netlist
X1 N3 N2 N2 Mout          ; NMOS Transistor Mout: Drain at N3, Gate at N2, Source/Body at N2
I1 N2 N1 DC 1A            ; Current Source I1 connected between VDD (N1) and N2
I2 N4 N5 DC 1A            ; Current Source Issu(t) connected between X (N4) and GND (N5)
CF N4 N2 1uF              ; Capacitor CF between X (N4) and N2
CL N3 N5 1uF              ; Capacitor CL between Vout (N3) and GND (N5)