spice
* SPICE netlist

V1 2 5 SIN(0 Vs AC_FREQ)  ; AC voltage source, where Vs is amplitude and AC_FREQ is the frequency
V2 5 0 VB                 ; DC voltage source providing bias
C1 2 0 C                  ; Capacitor
D1 4 3 DIODE_MODEL        ; Diode (assume a model DIODE_MODEL is defined elsewhere)